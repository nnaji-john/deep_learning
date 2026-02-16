# Johnpaul Nnaji
# University of The Cumberlands
# Statisics for Data Science (MSDS-531-M90)
# Drill with R on point estimates and confidence intervals
# 2/14//2026

# Load necessary libraries
# library(reticulate)
# use_virtualenv("/home/johnp/projects/deep_learning/.venv", required = TRUE)
# py_config()

# Question 1:
# load data
# Read the data from the website
chicago <- read.table("https://stat4ds.rwth-aachen.de/data/Chicago.dat", header = TRUE) # nolint

# View structure of the data
str(chicago)
summary(chicago$income)

# Descriptive graph (shape of the distribution)
# Histogram

hist(chicago$income,
    main = "Distribution of Income in Chicago",
    xlab = "Income (thousands of dollars)",
    ylab = "Frequency",
    col = "lightblue",
    border = "black"
)

# Boxplot for adding shape insight
boxplot(chicago$income,
    main = "Boxplot of Income in Chicago",
    ylab = "Income (thousandss of dollars)",
    col = "lightgreen",
    border = "black"
)
# Interpreetation from the histogram and boxplot.
# The histogram shows that the distribution of income in Chicago is right-skewed, with a long tail extending towards higher income values. This indicates that there are a few individuals with significantly higher incomes compared to the majority of the population. The boxplot confirms this observation by showing a longer whisker on the upper side, suggesting the presence of outliers or extreme values in the income data. Overall, the distribution of income in Chicago is not symmetric and is influenced by a small number of high-income earners. # nolint

# Point estimate (mean and standard deviation)
# Point estimate of population mean
mean_income <- mean(chicago$income)
mean_income

# Point estimate of population standard deviation
sd_income <- sd(chicago$income)
sd_income

# Interpretation of point estimates
# The point estimate of the population mean income in Chicago is approximately 20.33 thousand dollars, which suggests that, on average, individuals in Chicago earn around this amount. The point estimate of the population standard deviation is approximately 3.68 thousand dollars, indicating that there is some variability in income among individuals in Chicago. The standard deviation suggests that while many individuals earn close to the mean, there are also some who earn significantly more or less than the average income. # nolint

# 95% confidence interval for the population mean
# 95% confidence interval of mean
# Because n = 30 and the population standard deviation is unknown, we will use the t-distribution to calculate the confidence interval. # nolint
ci <- t.test(chicago$income, conf.level = 0.95)
ci$conf.int

# Interpretation of confidence interval
# The 95% confidence interval for the population mean income in Chicago is approximately (18.96k to 21.7k) thousand dollars, suggesting that we can be 95% confident that the true mean income of the population falls within this range. This interval provides a range of plausible values for the population mean income based on our sample data, and it indicates that the average income in Chicago is likely between 18.96k and 21.7k thousand dollars. # nolint



#Question 2:

# Load data from the website
anorexia <- read.table(("https://stat4ds.rwth-aachen.de/data/Anorexia.dat"), header = TRUE) # nolint
# View structure of the data
str(anorexia)
head(anorexia)

# Create weight change variable
anorexia$change <- anorexia$after - anorexia$before

# Create Variables for analysis
family <- subset(anorexia, therapy == "f")
nrow(family)

control <- subset(anorexia, therapy == "c")

# Descriptive statistic analysis
summary(family$change)
mean(family$change)
sd(family$change)
var(family$change)

# Graphical summaries
hist(family$change,
    breaks = 8,
    main = "Histogram:Weight Change (Family Therapy)",
    xlab = "Weight Change (lb)",
    ylab = "Frequency",
    col = "lightblue",
    border = "black"
)
boxplot(family$change,
    main = "Boxplot: Weight Change (Family Therapy)",
    ylab = "Weight Change (lb)",
    col = "lightgreen",
    border = "black"
)
qqnorm(family$change)
qqline(family$change, col = "red")

# Interpretation of the graphical summaries
# The histogram and boxplot indicates that weight change among the family therapy group are positive and roughly unimodal, with slight right skewness due to a few larger weight gains. The median and mean are both above zero, suggesting that participant recieving family therapy experiences overall weight increseas. The spread of the data indicates moderate variability and no severe outliers are observed. # nolint

# 95% confidence interval for difference in population means
t.test(family$change,
    control$change, # nolint
    conf.level = 0.95
)

# Interpretation of confidence interval
# Sample mean change (family):7.2647 lbs
# Sample mean change (control): -0.45 lbs
# Estinated mean difference: 7.7147 lbs
# the 95% confidence interval:(2.9766, 12.4528) lbs
# Interpretation: We are 95% confident that the population mean weight change for girls receiving familiy therapy is between 2.98 and 12.45 pounds greater than the population mean weight change for the girls in the control group. Thus,  Family therapy shows a significantly higher mean weight gain than control(Welch t-test: t = 3.29, df = 36.98, p =0.0022). # nolint