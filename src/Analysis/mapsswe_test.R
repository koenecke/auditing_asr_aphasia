# Load necessary package
library(dplyr)

# These functions execute the MAPPSWE tests based on (test statistic based on 
##Gillick and Cox (1989) and 
##https://web.stanford.edu/~jurafsky/slp3/16.pdf#page=16.52)

# input argument : data frame, variable group 1 name, variable group 2 name 
# output argument: data frame of result 
# use when it is a wide data frame 
mapsswe_test <- function(df, var1, var2) {
  # Check if the columns exist in the dataframe
  if (!(var1 %in% names(df)) | !(var2 %in% names(df))) {
    stop("Error: One or both variable names do not exist in the dataframe.")
  }
  
  # Extract the two columns as variables
  x <- df[[var1]]
  y <- df[[var2]]
  
  # Calculate the difference between the two variables
  diff_x_y <- x - y
  # calculate average of difference
  mu_hat <- mean(diff_x_y)
  # calculate variance of difference
  var_hat <- var(diff_x_y)
  # calculate sd of difference
  sd_hat <- sqrt(var_hat)
  # calculate sample size n 
  n <- length(diff_x_y)
  
  # Calculate the W value 
  w_value <- mu_hat / (sd_hat / sqrt(n))
  
  # Compute one-tail p-value
  if (w_value < 0) {
    # Left-tail test
    p_val_one_tail <- pnorm(w_value, mean = 0, sd = 1, lower.tail = TRUE)
  } else {
    # Right-tail test
    p_val_one_tail <- pnorm(w_value, mean = 0, sd = 1, lower.tail = FALSE)
  }
  
  # Compute two-tailed p-value (using absolute value of W)
  p_val_two_tail <- 2 * pnorm(abs(w_value), mean = 0, sd = 1, lower.tail = FALSE)
  
  # Return the results as a list
  result_df <- data.frame(
    W_Statistic = w_value,
    P_value_One_Tail = p_val_one_tail,
    P_value_Two_Tail = p_val_two_tail,
    Mean_Difference = mu_hat,
    Variance = var_hat,
    Sample_Size = n
  )
  
  return(result_df)
}



# use for long format data frame and more than two group comparison 
pairwise_mapsswe_test <- function(df, numeric_var, group_var) {
  # Check if the columns exist in the dataframe
  if (!(numeric_var %in% names(df)) | !(group_var %in% names(df))) {
    stop("Error: One or both variable names do not exist in the dataframe.")
  }
  
  # Get all unique levels in the grouping variable
  group_levels <- unique(df[[group_var]])
  
  # Check if there are at least two levels
  if (length(group_levels) < 2) {
    stop("Error: Grouping variable must have at least two levels.")
  }
  
  # Initialize an empty list to store results for each pairwise comparison
  results_list <- list()
  
  # Perform pairwise comparisons
  for (i in 1:(length(group_levels) - 1)) {
    for (j in (i + 1):length(group_levels)) {
      group1_name <- group_levels[i]
      group2_name <- group_levels[j]
      
      # Filter the data for the two groups
      group1 <- df %>% filter(df[[group_var]] == group1_name) %>% pull(numeric_var)
      group2 <- df %>% filter(df[[group_var]] == group2_name) %>% pull(numeric_var)
      
      # Ensure the two groups have data
      if (length(group1) == 0 | length(group2) == 0) {
        stop("Error: One of the groups has no data.")
      }
      
      # Calculate the difference between the two groups
      diff_x_y <- group1 - group2[1:length(group1)] # Assumes equal sample sizes; adjust for unequal
      
      # Mean, variance, standard deviation, and sample size
      mu_hat <- mean(diff_x_y)
      var_hat <- var(diff_x_y)
      sd_hat <- sqrt(var_hat)
      n <- length(diff_x_y)
      
      # Calculate the W value (test statistic)
      w_value <- mu_hat / (sd_hat / sqrt(n))
      
      # Compute one-tail p-value
      if (w_value < 0) {
        p_val_one_tail <- round(pnorm(w_value, mean = 0, sd = 1, lower.tail = TRUE),digits = 4)
      } else {
        p_val_one_tail <- round(pnorm(w_value, mean = 0, sd = 1, lower.tail = FALSE),digits = 4)
      }
      
      # Compute two-tailed p-value (using absolute value of W)
      p_val_two_tail <- round(2 * pnorm(abs(w_value), mean = 0, sd = 1, lower.tail = FALSE),digits = 4)
      
      # Store results for this pairwise comparison
      results_list[[paste(group1_name, "vs", group2_name)]] <- data.frame(
        Group1 = group1_name,
        Group2 = group2_name,
        W_Statistic = w_value,
        P_value_One_Tail = p_val_one_tail,
        P_value_Two_Tail = p_val_two_tail,
        Mean_Difference = mu_hat,
        Variance = var_hat,
        Sample_Size = n
      )
    }
  }
  
  # Combine the individual pairwise results into one dataframe
  result_df <- bind_rows(results_list, .id = "Comparison")
  
  return(result_df)
}

# Example usage:
# Assuming your dataframe is called `df`, with a numeric variable 'x' and a grouping variable 'group'
# result <- pairwise_mapsswe_test(df, "x", "group")
# print(result)
