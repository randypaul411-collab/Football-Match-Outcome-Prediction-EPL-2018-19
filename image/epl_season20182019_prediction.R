# ============================================================
# Football Match Outcome Prediction using Poisson Regression
# English Premier League 2024/25 Season
# Author: Paul Agbekpornu
# Tools: R, dplyr, ggplot2, tidyr, boot
# Data Source: football-data.co.uk (free open data)
# ============================================================

# --- 1. Load Libraries ---
library(dplyr)
library(tidyr)
library(ggplot2)
library(boot)
library(gridExtra)

# --- 2. Load Data ---
# Data from: https://www.football-data.co.uk/englandm.php
# EPL 2018/19 season match results
url <- "https://www.football-data.co.uk/mmz4281/1819/E0.csv"
epl <- tryCatch(
  read.csv(url, stringsAsFactors = FALSE),
  error = function(e) {
    message("Using local fallback data")
    NULL
  }
)


# Fallback: generate realistic synthetic EPL data for demonstration
if (is.null(epl) || nrow(epl) < 10) {
  set.seed(42)
  teams <- c("Arsenal","Manchester City","Liverpool","Chelsea","Tottenham",
             "Manchester Utd","Newcastle","Crystal Palace","Brighton","Leicester")
  n_games <- 200
  home <- sample(teams, n_games, replace = TRUE)
  away <- sample(teams, n_games, replace = TRUE)
  # Remove same-team matchups
  valid <- home != away
  home <- home[valid][1:180]
  away <- away[valid][1:180]
  # Poisson goals with home advantage
  home_goals <- rpois(180, lambda = 1.5)
  away_goals <- rpois(180, lambda = 1.1)
  epl <- data.frame(
    HomeTeam = home,
    AwayTeam = away,
    FTHG = home_goals,
    FTAG = away_goals,
    stringsAsFactors = FALSE
  )
}

# --- 3. Clean and Prepare Data ---
matches <- epl %>%
  select(HomeTeam, AwayTeam, FTHG, FTAG) %>%
  filter(!is.na(FTHG), !is.na(FTAG)) %>%
  mutate(
    Result = case_when(
      FTHG > FTAG ~ "Home Win",
      FTHG < FTAG ~ "Away Win",
      TRUE ~ "Draw"
    ),
    TotalGoals = FTHG + FTAG
  )

cat("=== Dataset Overview ===\n")
cat("Total matches:", nrow(matches), "\n")
cat("Teams:", length(unique(c(matches$HomeTeam, matches$AwayTeam))), "\n\n")

# --- 4. Descriptive Statistics ---
cat("=== Match Result Distribution ===\n")
result_table <- table(matches$Result)
print(result_table)
cat("\nHome Win Rate:", round(mean(matches$Result == "Home Win") * 100, 1), "%\n")
cat("Draw Rate:", round(mean(matches$Result == "Draw") * 100, 1), "%\n")
cat("Away Win Rate:", round(mean(matches$Result == "Away Win") * 100, 1), "%\n\n")

cat("=== Goals Statistics ===\n")
cat("Average Home Goals:", round(mean(matches$FTHG), 2), "\n")
cat("Average Away Goals:", round(mean(matches$FTAG), 2), "\n")
cat("Average Total Goals per Match:", round(mean(matches$TotalGoals), 2), "\n\n")

# --- 5. Team Attack & Defence Strength ---
home_attack <- matches %>%
  group_by(Team = HomeTeam) %>%
  summarise(HomeGoalsScored = mean(FTHG), HomeGoalsConceded = mean(FTAG))

away_attack <- matches %>%
  group_by(Team = AwayTeam) %>%
  summarise(AwayGoalsScored = mean(FTAG), AwayGoalsConceded = mean(FTHG))

team_strength <- home_attack %>%
  left_join(away_attack, by = "Team") %>%
  mutate(
    AttackStrength = (HomeGoalsScored + AwayGoalsScored) / 2,
    DefenceStrength = (HomeGoalsConceded + AwayGoalsConceded) / 2
  ) %>%
  arrange(desc(AttackStrength))

cat("=== Top 5 Teams by Attack Strength ===\n")
print(head(team_strength[, c("Team", "AttackStrength", "DefenceStrength")], 5))
cat("\n")

# --- 6. Poisson Regression Model ---
# Convert to long format for Poisson GLM
home_data <- matches %>%
  select(Team = HomeTeam, Opponent = AwayTeam, Goals = FTHG) %>%
  mutate(Home = 1)

away_data <- matches %>%
  select(Team = AwayTeam, Opponent = HomeTeam, Goals = FTAG) %>%
  mutate(Home = 0)

model_data <- bind_rows(home_data, away_data)

# Fit Poisson GLM -- classical interpretable model
poisson_model <- glm(Goals ~ Home + Team + Opponent,
                     data = model_data,
                     family = poisson(link = "log"))

cat("=== Poisson GLM Model Summary ===\n")
cat("AIC:", round(AIC(poisson_model), 2), "\n")
cat("Null Deviance:", round(poisson_model$null.deviance, 2), "\n")
cat("Residual Deviance:", round(poisson_model$deviance, 2), "\n")
cat("McFadden R²:", round(1 - (poisson_model$deviance / poisson_model$null.deviance), 4), "\n\n")

# Home advantage coefficient
home_coef <- coef(poisson_model)["Home"]
cat("Home Advantage (log scale):", round(home_coef, 4), "\n")
cat("Home Advantage (multiplicative):", round(exp(home_coef), 4),
    "-- teams score", round((exp(home_coef) - 1) * 100, 1), "% more goals at home\n\n")

# --- 7. Predict Match Outcome Function ---
predict_match <- function(home_team, away_team, model, n_simulations = 10000) {
  # Predict expected goals using Poisson model
  new_home <- data.frame(Team = home_team, Opponent = away_team, Home = 1)
  new_away <- data.frame(Team = away_team, Opponent = home_team, Home = 0)

  lambda_home <- tryCatch(
    predict(model, newdata = new_home, type = "response"),
    error = function(e) mean(matches$FTHG)
  )
  lambda_away <- tryCatch(
    predict(model, newdata = new_away, type = "response"),
    error = function(e) mean(matches$FTAG)
  )

  # Monte Carlo simulation using Poisson distribution
  set.seed(123)
  home_goals_sim <- rpois(n_simulations, lambda_home)
  away_goals_sim <- rpois(n_simulations, lambda_away)

  home_win_prob <- mean(home_goals_sim > away_goals_sim)
  draw_prob <- mean(home_goals_sim == away_goals_sim)
  away_win_prob <- mean(home_goals_sim < away_goals_sim)

  # Bootstrap confidence intervals on home win probability
  boot_fn <- function(data, indices) {
    d <- data[indices, ]
    mean(d$home > d$away)
  }
  sim_data <- data.frame(home = home_goals_sim, away = away_goals_sim)
  boot_result <- boot(sim_data, boot_fn, R = 1000)
  ci <- boot.ci(boot_result, type = "perc", conf = 0.95)

  list(
    HomeTeam = home_team,
    AwayTeam = away_team,
    ExpectedHomeGoals = round(lambda_home, 2),
    ExpectedAwayGoals = round(lambda_away, 2),
    HomeWinProb = round(home_win_prob * 100, 1),
    DrawProb = round(draw_prob * 100, 1),
    AwayWinProb = round(away_win_prob * 100, 1),
    HomeWinCI_Lower = round(ci$percent[4] * 100, 1),
    HomeWinCI_Upper = round(ci$percent[5] * 100, 1)
  )
}

# --- 8. Sample Predictions ---
teams_available <- unique(model_data$Team)
cat("=== Match Outcome Predictions ===\n\n")

# Pick 3 matchups from available teams
matchups <- list(
  c(teams_available[1], teams_available[2]),
  c(teams_available[3], teams_available[4]),
  c(teams_available[5], teams_available[1])
)

for (m in matchups) {
  result <- predict_match(m[1], m[2], poisson_model)
  cat(sprintf("%-20s vs %-20s\n", result$HomeTeam, result$AwayTeam))
  cat(sprintf("  Expected Goals: %.2f - %.2f\n",
              result$ExpectedHomeGoals, result$ExpectedAwayGoals))
  cat(sprintf("  Home Win: %s%%  |  Draw: %s%%  |  Away Win: %s%%\n",
              result$HomeWinProb, result$DrawProb, result$AwayWinProb))
  cat(sprintf("  Home Win 95%% CI: [%s%%, %s%%]\n\n",
              result$HomeWinCI_Lower, result$HomeWinCI_Upper))
}

# --- 8. Sample Predictions ---
teams_available <- unique(model_data$Team)
cat("=== Match Outcome Predictions ===\n\n")

matchups <- list(
  c(teams_available[1], teams_available[2]),
  c(teams_available[3], teams_available[4]),
  c(teams_available[5], teams_available[1])
)

# Collect results into a list for Plot 4
all_results <- list()

for (m in matchups) {
  result <- predict_match(m[1], m[2], poisson_model)
  all_results[[length(all_results) + 1]] <- result
  cat(sprintf("%-20s vs %-20s\n", result$HomeTeam, result$AwayTeam))
  cat(sprintf("  Expected Goals: %.2f - %.2f\n",
              result$ExpectedHomeGoals, result$ExpectedAwayGoals))
  cat(sprintf("  Home Win: %s%%  |  Draw: %s%%  |  Away Win: %s%%\n",
              result$HomeWinProb, result$DrawProb, result$AwayWinProb))
  cat(sprintf("  Home Win 95%% CI: [%s%%, %s%%]\n\n",
              result$HomeWinCI_Lower, result$HomeWinCI_Upper))
}

# --- 9. Visualizations ---

# Plot 1: Result Distribution
p1 <- ggplot(matches, aes(x = Result, fill = Result)) +
  geom_bar(color = "white", width = 0.6) +
  scale_fill_manual(values = c("Home Win" = "#1B4F8A",
                                "Draw" = "#888888",
                                "Away Win" = "#C0392B")) +
  labs(title = "Premier League Match Result Distribution",
       subtitle = paste0("Based on ", nrow(matches), " matches"),
       x = "Result", y = "Number of Matches") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold"))

ggsave("C:/Users/User/Documents/sport/plot1_result_distribution.png",
       p1, width = 8, height = 5, dpi = 150)

# Plot 2: Goals Distribution -- Poisson fit
goals_data <- data.frame(
  Goals = c(matches$FTHG, matches$FTAG),
  Type = c(rep("Home", nrow(matches)), rep("Away", nrow(matches)))
)

p2 <- ggplot(goals_data, aes(x = Goals, fill = Type)) +
  geom_histogram(binwidth = 1, position = "dodge",
                 color = "white", alpha = 0.85) +
  scale_fill_manual(values = c("Home" = "#1B4F8A", "Away" = "#C0392B")) +
  labs(title = "Goals Scored Distribution -- Home vs Away",
       subtitle = "Poisson-distributed goal scoring rates",
       x = "Goals Scored", y = "Frequency", fill = "Team") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("C:/Users/User/Documents/sport/plot2_goals_distribution.png",
       p2, width = 8, height = 5, dpi = 150)

# Plot 3: Team Attack vs Defence
p3 <- ggplot(team_strength, aes(x = DefenceStrength, y = AttackStrength,
                                  label = Team)) +
  geom_point(size = 3, color = "#1B4F8A", alpha = 0.8) +
  geom_text(vjust = -0.8, size = 3, color = "#333333") +
  geom_hline(yintercept = mean(team_strength$AttackStrength),
             linetype = "dashed", color = "#888888") +
  geom_vline(xintercept = mean(team_strength$DefenceStrength),
             linetype = "dashed", color = "#888888") +
  labs(title = "Team Attack vs Defence Strength",
       subtitle = "Quadrant analysis: top-left = strong attack, strong defence",
       x = "Goals Conceded per Game (lower = better defence)",
       y = "Goals Scored per Game (higher = better attack)") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("C:/Users/User/Documents/sport/plot3_team_strength.png",
       p3, width = 9, height = 6, dpi = 150)

# Plot 4: Outcome Probability by Fixture -- Stacked Horizontal Bars
# Requires predictions from predict_match() function

pred_df <- data.frame(
  fixture = vapply(all_results, function(r) paste(r$HomeTeam, "vs", r$AwayTeam), character(1)),
  hw      = vapply(all_results, function(r) as.numeric(r$HomeWinProb), numeric(1)),
  draw    = vapply(all_results, function(r) as.numeric(r$DrawProb),    numeric(1)),
  aw      = vapply(all_results, function(r) as.numeric(r$AwayWinProb), numeric(1)),
  stringsAsFactors = FALSE
)

stacked_df <- pred_df %>%
  pivot_longer(cols = c(hw, draw, aw),
               names_to  = "outcome",
               values_to = "prob") %>%
  mutate(
    outcome = factor(outcome,
                     levels = c("hw", "draw", "aw"),
                     labels = c("Home Win", "Draw", "Away Win")),
    fixture = factor(fixture, levels = rev(pred_df$fixture))
  )

label_df <- pred_df %>%
  mutate(
    fixture  = factor(fixture, levels = rev(pred_df$fixture)),
    hw_mid   = hw / 2,
    draw_mid = hw + draw / 2,
    aw_mid   = hw + draw + aw / 2
  )

p4 <- ggplot(stacked_df, aes(y = fixture, x = prob, fill = outcome)) +
  geom_col(width = 0.55, colour = "white", linewidth = 0.5) +
  geom_vline(xintercept = 50, linetype = "dashed",
             colour = "black", linewidth = 0.6, alpha = 0.5) +
  geom_text(data = label_df,
            aes(y = fixture, x = hw_mid,
                label = paste0(round(hw, 1), "%")),
            inherit.aes = FALSE,
            colour = "white", fontface = "bold", size = 3.2) +
  geom_text(data = label_df,
            aes(y = fixture, x = draw_mid,
                label = paste0(round(draw, 1), "%")),
            inherit.aes = FALSE,
            colour = "white", fontface = "bold", size = 3.2) +
  geom_text(data = label_df,
            aes(y = fixture, x = aw_mid,
                label = paste0(round(aw, 1), "%")),
            inherit.aes = FALSE,
            colour = "white", fontface = "bold", size = 3.2) +
  scale_fill_manual(values = c(
    "Home Win" = "#1B4F8A",
    "Draw"     = "#888888",
    "Away Win" = "#C0392B"
  )) +
  scale_x_continuous(labels = function(x) paste0(x, "%"),
                     limits = c(0, 100),
                     expand = expansion(mult = c(0, 0.01))) +
  labs(title    = "Outcome Probability by Fixture",
       subtitle = "Monte Carlo simulation (Poisson) | 50,000 simulations per match",
       x        = "Probability (%)",
       y        = NULL,
       fill     = NULL) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold"),
    plot.subtitle      = element_text(size = 9, colour = "grey40"),
    legend.position    = "bottom",
    panel.grid.major.y = element_blank()
  )

ggsave("C:/Users/User/Documents/sport/plot4_outcome_probability.png",
       p4, width = 9, height = 6, dpi = 150)


# Plot 4: Outcome Probability by Fixture -- Stacked Horizontal Bars
pred_df <- data.frame(
  fixture = sapply(all_results, function(r) paste(r$HomeTeam, "vs", r$AwayTeam)),
  hw      = sapply(all_results, `[[`, "HomeWinProb"),
  draw    = sapply(all_results, `[[`, "DrawProb"),
  aw      = sapply(all_results, `[[`, "AwayWinProb"),
  stringsAsFactors = FALSE
)

# Long format for stacked bars
stacked_df <- pred_df %>%
  pivot_longer(
    cols      = c(hw, draw, aw),
    names_to  = "outcome",
    values_to = "prob"
  ) %>%
  mutate(
    outcome = factor(outcome,
                     levels = c("aw", "draw", "hw"),   # aw first = left side
                     labels = c("Away Win", "Draw", "Home Win")),
    fixture = factor(fixture, levels = rev(pred_df$fixture))
  )

# Label midpoints for each segment
label_df <- pred_df %>%
  mutate(
    fixture  = factor(fixture, levels = rev(pred_df$fixture)),
    hw_mid   = hw / 2,              # Home Win midpoint -- starts at 0
    draw_mid = hw + (draw / 2),     # Draw midpoint -- starts after hw
    aw_mid   = hw + draw + (aw / 2) # Away Win midpoint -- starts after hw + draw
  )

p4 <- ggplot(stacked_df, aes(y = fixture, x = prob, fill = outcome)) +
  geom_col(width = 0.55, colour = "white", linewidth = 0.5) +
  geom_vline(xintercept = 50, linetype = "dashed",
             colour = "black", linewidth = 0.6, alpha = 0.5) +
  # Away Win label
  geom_text(data = label_df,
            aes(y = fixture, x = aw_mid, label = paste0(aw, "%")),
            inherit.aes = FALSE,
            colour = "white", fontface = "bold", size = 3.5) +
  # Draw label
  geom_text(data = label_df,
            aes(y = fixture, x = draw_mid, label = paste0(draw, "%")),
            inherit.aes = FALSE,
            colour = "white", fontface = "bold", size = 3.5) +
  # Home Win label
  geom_text(data = label_df,
            aes(y = fixture, x = hw_mid, label = paste0(hw, "%")),
            inherit.aes = FALSE,
            colour = "white", fontface = "bold", size = 3.5) +
  scale_fill_manual(values = c(
    "Away Win" = "#C0392B",   # red on left
    "Draw"     = "#888888",   # grey in middle
    "Home Win" = "#1B4F8A"    # blue on right
  )) +
  scale_x_continuous(
    labels = function(x) paste0(x, "%"),
    limits = c(0, 100),
    expand = expansion(mult = c(0, 0.01))
  ) +
  labs(
    title    = "Outcome Probability by Fixture",
    subtitle = "Monte Carlo Poisson simulation -- 10,000 simulations per match",
    x        = "Probability (%)",
    y        = NULL,
    fill     = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title         = element_text(face = "bold"),
    plot.subtitle      = element_text(size = 9, colour = "grey40"),
    legend.position    = "bottom",
    panel.grid.major.y = element_blank()
  )

ggsave("C:/Users/User/Documents/sport/plot4_outcome_probability.png",
       p4, width = 9, height = 6, dpi = 150)

#Dashboard
# ============================================================
# COMBINE ALL 4 INTO ONE DASHBOARD
# ============================================================
dashboard <- arrangeGrob(
  p1, p2, p3, p4,
  ncol = 2,
  top  = grid::textGrob(
    "Football Match Outcome Prediction\nPoisson GLM + Bootstrap CI | EPL 2024/25",
    gp = grid::gpar(fontsize = 14, fontface = "bold")
  )
)

ggsave(
  "C:/Users/User/Documents/sport/2018_19_match_prediction_dashboard.png",
  dashboard,
  width  = 16,
  height = 12,
  dpi    = 150
)

cat("Dashboard saved.\n")


cat("=== Model Diagnostics ===\n")
cat("Dispersion check (should be ~1 for good Poisson fit):",
    round(poisson_model$deviance / poisson_model$df.residual, 3), "\n")
cat("Plots saved to C:/Users/User/Documents/sport/football_prediction/\n")
cat("\n=== Project Complete ===\n")
