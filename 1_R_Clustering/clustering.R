smokers <- matrix(
  c(36, 168, 50, 0.7, 8, 20, 15, 16, 12,
    41, 170, 64, 0.7, 8, 20, 20, 21, 21,
    35, 169, 65, 0.8, 10, 20, 30, 15, 23,
    55, 177, 68, 1.0, 12, 20, 20, 35, 35,
    21, 172, 80, 0.7, 10, 16, 15, 5, 4,
    46, 165, 53, 1.4, 17, 20, 20, 26, 26,
    38, 167, 57, 0.1, 1, 20, 10, 18, 9,
    37, 165, 58, 0.1, 1, 20, 20, 17, 17,
    40, 156, 53, 0.4, 5, 18, 20, 22, 22,
    36, 172, 55, 0.5, 6, 20, 20, 16, 16,
    39, 168, 65, 0.7, 8, 16, 10, 23, 12), nrow=11, ncol=9, byrow = TRUE)
colnames(smokers) <- c('Age','Height','Weight', 'Nicotine',
                       'Tar', 'AgeSt', 'CPD', 'YearS', 'PY')
rownames(smokers) <- c(1,2,3,4,5,6,7,8,9,10,11)

library(stats)
set.seed(1234)
cl<-kmeans(smokers, centers=4)
cl

cl$size
cl$centers
cl$cluster
cl$withinss
cl$betweenss/cl$totss
cl$betweenss
cl$totss
cl$tot.withinss

#cluster 1 - within-cluster sum of squares by cluster
xm1=array(0)
for(m in 1:9)
{
  xm1[m]= sum(smokers[c('1','7','8','10','11'),m])  / 5
}
xm1

r1.wc=array(0)
for(m in 1:9)
{
  r1.wc[m]= sum( (smokers[c('1','7','8','10','11'),m]  - xm1[m])^2 )
}
sum(r1.wc)

#cluster 2 - within-cluster sum of squares by cluster
xm2=array(0)
for(m in 1:9)
{
  xm2[m]= sum(smokers[c('2','3','6','9'),m])  / 4
}

r2.wc=array(0)
for(m in 1:9)
{
  r2.wc[m]= sum( (smokers[c('2','3','6','9'),m]  - xm2[m])^2 )
}
sum(r2.wc)

#between-cluster sum of squares
btwss.r1=btwss.r2=btwss.r3=btwss.r4=btwss.r5=array(0)

#r=1
for(m in 1:9) btwss.r1[m]=5* ( ( ( sum(smokers[c('1','7','8','10','11'),m])  / 5)  - mean(smokers[,m]) )^2 )
sum.btwss.r1=sum(btwss.r1)

#r=2
for(m in 1:9) btwss.r2[m]=4*( ( ( sum(smokers[c('2','3','6','9'),m])  / 4)  - mean(smokers[,m]) )^2 )
sum.btwss.r2=sum(btwss.r2)

#r=3
for(m in 1:9) btwss.r3[m]=( ( smokers[5,m] ) - mean(smokers[,m]) )^2
sum.btwss.r3=sum(btwss.r3)

#r=4
for(m in 1:9) btwss.r4[m]=( ( smokers[4,m] ) - mean(smokers[,m]) )^2
sum.btwss.r4=sum(btwss.r4)
sum(sum.btwss.r1,sum.btwss.r2,sum.btwss.r3,sum.btwss.r4)

#total sum of squares
tts=0
for(m in 1:9)
{
  for (i in 1:11)
  {
      tts=tts+(smokers[i,m]-mean(smokers[,m]))^2
  }
}
tts

#within-cluster sum of squares for different number of clusters
wss=array(0)
for (r in 1:8) wss[r] <- kmeans(smokers,centers=r)$tot.withinss
plot(1:8, wss, type='b', xlab='Number of clusters',ylab='Within-cluster sum of squares')

# ==============================================================================
# The upgraded version of the clustering uses Silhouette coefficient, which
# measure how similar an object is to its own cluster compared to other clusters
# for an observation i: 
#s(i)= b(i)-q(i)/max{a(i),b(i)} 
#where a(i) is the mean distance to other points in the same cluster
#b(i) is the mean distance to then nearest neighboring cluster
#s(i): the range ferom -1 to 1
# ==============================================================================
library(tidyverse)
library(cluster)      # For silhouette analysis
library(factoextra)   # For beautiful, publication-ready clustering plots
library(broom)        # For tidying model outputs

# 1. DATA INGESTION & ACTUARIAL FEATURE ENGINEERING
# -----------------------------------------------------------------------------
smokers_df_new <- data.frame(
  Age = c(36,41,35,55,21,46,38,37,40,36,39),
  Height = c(168,170,169,177,172,165,167,165,156,172,168),
  Weight = c(50,64,65,68,80,53,57,58,53,55,65),
  Nicotine = c(0.7,0.7,0.8,1.0,0.7,1.4,0.1,0.1,0.4,0.5,0.7),
  Tar = c(8,8,10,12,10,17,1,1,5,6,8),
  AgeSt = c(20,20,20,20,16,20,20,20,18,20,16),
  CPD = c(15,20,30,20,15,20,10,20,20,20,10),
  YearS = c(16,21,15,35,5,26,18,17,22,16,23)
  # Note: 'PY' (Pack Years) is intentionally dropped to prevent deterministic 
  # multicollinearity, as PY = (CPD / 20) * YearS.
)

# 2. STANDARDIZATION (Crucial for Distance-Based Algorithms)
# -----------------------------------------------------------------------------
# Math: z = (x - mu) / sigma. Prevents variables with large ranges (Height) 
# from dominating variables with small ranges (Nicotine).
smokers_new_scaled <- scale(smokers_df_new)

# 3. DETERMINING OPTIMAL 'K' (Silhouette & Gap Statistic)
# -----------------------------------------------------------------------------
# We use the Silhouette method to mathematically validate cluster separation.
fviz_nbclust(smokers_new_scaled, kmeans, method = "silhouette", k.max = 6) +
  labs(title = "Actuarial Risk Segmentation: Silhouette Optimization",
       subtitle = "Identifying distinct underwriting classes") +
  theme_minimal()

# Let's assume k=3 is optimal based on the plot (distinct risk tiers)
set.seed(1234) # For reproducibility in audits
kmeans_model <- kmeans(smokers_new_scaled, centers = 3, nstart = 25) 
# nstart=25 ensures we don't get stuck in a local minimum (a common actuarial modeling risk)

# 4. EVALUATING CLUSTER QUALITY (Silhouette Plot)
# -----------------------------------------------------------------------------
sil_stats <- silhouette(kmeans_model$cluster, dist(smokers_new_scaled))
fviz_silhouette(sil_stats) +
  labs(title = "Cluster Homogeneity Check",
       subtitle = "Positive values indicate correct risk classification")

# 5. INTERPRETING THE RISK TIERS (Actuarial Business Logic)
# -----------------------------------------------------------------------------
# We must map the mathematical clusters back to business reality.
# What do these clusters mean for Pricing or Underwriting?
#That is the most important question that we want to answer

cluster_profiles <- smokers_df_new %>%
  mutate(Cluster = as.factor(kmeans_model$cluster)) %>%
  group_by(Cluster) %>%
  summarise(across(everything(), mean)) %>%
  ungroup()

print(cluster_profiles)

#==================================================================
# we can make the silhouette score jump if we reduced "noise"
# the most critical drivers of risk are age CPD(cigarettes per day)
# and weight because it is an independent variable that give us a
# picture for morbidity risk
# removing height,tar,nicotine and ageSt(age started) will make the model more 
# accurate. 
#==================================================================
smokers_new_scaledv1<- scale(smokers_df_new %>% select(Age, CPD, Weight))
#K-Means on the new cleaner data
set.seed(1234)
kmeans_model_v1 <- kmeans(smokers_new_scaledv1, centers = 3, nstart = 25)

# 2. Check the new Silhouette Score
sil_stats_v1 <- silhouette(kmeans_model_v1$cluster, dist(smokers_new_scaledv1))
fviz_silhouette(sil_stats_v1) +
  labs(title = "Improved Cluster Homogeneity (Age, CPD, Weight)",
       subtitle = "Notice how much higher the average silhouette width is now!")

#Printing the new cluster profiles to see the business logic
cluster_profiles_v1 <- smokers_df_new %>%
  mutate(Cluster = as.factor(kmeans_model_v1$cluster)) %>%
  group_by(Cluster) %>%
  summarise(across(everything(), mean)) %>%
  ungroup()

print(cluster_profiles_v1)

#=========================================================================
# as we can observe for the v1 model. the 3rd cluster have an outlier
# we need to remove the row 5: 21-year-old weighting 80kg. 
#=========================================================================

smokers_v2 <- smokers_df_new[-5, ] #we are removing the ourlier here 

# Scaling the v2 data (Using our selected features: Age, CPD, Weight)
smokers_v2_scaled <- scale(smokers_v2 %>% select(Age, CPD, Weight))

set.seed(1234)
kmeans_v2 <- kmeans(smokers_v2_scaled, centers = 3, nstart = 25)

sil_v2 <- silhouette(kmeans_v2$cluster, dist(smokers_v2_scaled))
fviz_silhouette(sil_v2) +
  labs(title = "Model Iteration v2: Outlier Removed (k=3)",
       subtitle = "Checking for improved homogeneity and balanced clusters")

cluster_profiles_v2 <- smokers_v2 %>%
  mutate(Cluster = as.factor(kmeans_v2$cluster)) %>%
  group_by(Cluster) %>%
  summarise(across(everything(), mean)) %>%
  ungroup()

print(cluster_profiles_v2)

#==================================================================
#the data told us the reality which is k=3 is not suitable for our
#small scale analysis so we will change the k from 3 to 2 and observe
#the changes in our 3rd iteration 
#A two-tier rating structure (Standard vs. Substandard) 
#===================================================================

smokers_v3 <- smokers_v2 # we are keeping the clean data from v2

smokers_v3_scaled <- scale(smokers_v3 %>% select(Age, CPD, Weight))

# we are going to  Run the  K-Means with k=2 (Letting the data dictate the structure)
set.seed(1234)
kmeans_v3 <- kmeans(smokers_v3_scaled, centers = 2, nstart = 25)

#Checking the Silhouette Score for k=2
sil_v3 <- silhouette(kmeans_v3$cluster, dist(smokers_v3_scaled))
fviz_silhouette(sil_v3) +
  labs(title = "Model Iteration v3: Two-Tier Risk Structure (k=2)",
       subtitle = "Testing a Standard vs. Substandard underwriting split")
#Printing the final profiles
cluster_profiles_v3 <- smokers_v3 %>%
  mutate(Cluster = as.factor(kmeans_v3$cluster)) %>%
  group_by(Cluster) %>%
  summarise(across(everything(), mean)) %>%
  ungroup()
print(cluster_profiles_v3)

#===============================================================================
# due to the lack of data we can't improve the model more we need to 
# find more data so we can have better outcomes 
#In multivariate statistics, a common rule of thumb is that we 
#need at least 10 to 20 observations per variable to achieve stable clustering
#we had 3 variables, meaning we ideally need 30 to 60 data points
#===============================================================================

# we found a similar structured dataset from kaggle done by Africa 
#Health Lab "Smoking Health Dataset" that we're gonna use in our 
#4th iteration. 
smokers_v4<- read.csv("health_dataset.csv", stringsAsFactors = FALSE)
View(smokers_v4)
# now we need to explore this dataset before running any analysis 
#EDA phase 
library(tidyverse)
#Basic Structure and Dimensions
cat("--- DATASET DIMENSIONS ---\n")
cat("Rows:", nrow(smokers_v4), "Columns:", ncol(smokers_v4), "\n\n")
cat("--- DATA STRUCTURE ---\n")
str(smokers_v4)
cat("\n")
#Checking for Missing Values 
cat("--- MISSING VALUES CHECK ---\n")
missing_counts <- colSums(is.na(smokers_v4))
print(missing_counts[missing_counts > 0])
if(sum(missing_counts) == 0) {
  cat("Great news: No missing values in the dataset!\n\n")
} else {
  cat("Warning: There are missing values that need to be handled.\n\n")
}
# her we're gonna Examine the Target Population: Smoking Status
cat("--- SMOKING STATUS DISTRIBUTION ---\n")
print(table(smokers_v4$Smoking_Status))
cat("\n")
#Statistical Summary of Key Continuous Variables
#We care most about Age, Cigarettes_Per_Day, and BMI which represent the weight
# but in a very standardized way it normalizes weight by height.
cat("--- SUMMARY OF KEY RISK VARIABLES ---\n")
summary(smokers_raw %>% select(Age, Cigarettes_Per_Day, BMI))
cat("\n")
# Quick Visualizations (Histograms) to spot skewness or outliers
# This helps us understand if we need to worry about extreme outliers before clustering
smokers_v4 %>%
  select(Age, Cigarettes_Per_Day, BMI) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.8) +
  facet_wrap(~ Variable, scales = "free_x") +
  labs(title = "Distribution of Key Actuarial Risk Variables",
       subtitle = "Checking for skewness and potential outliers before modeling",
       x = "Value",
       y = "Frequency"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

#===============================================================================
# if we made the analysis with nonsmokers without separating them we're 
# gonna skew the results due to their weight in CPD so filtering the 
#data to look only at current smokers, and then look at the distributions again
#is a better approach.
#===============================================================================
smokers_current <- smokers_v4 %>%
  filter(Smoking_Status == "Current")

cat("We are now analyzing", nrow(smokers_current), "active smokers.\n\n")

#Re-plotting the histograms for the filtered data
smokers_current %>%
  select(Age, Cigarettes_Per_Day, BMI) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "darkred", color = "white", alpha = 0.8) +
  facet_wrap(~ Variable, scales = "free_x") +
  labs(
    title = "Distribution of Key Actuarial Risk Variables (Current Smokers Only)",
    subtitle = "Now we can see the true shape of the smoker risk profile",
    x = "Value",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# ==============================================================================
# smoker_v4: RISK SEGMENTATION FOR CURRENT SMOKERS
# we're gonna run the 4th iteration 
# ==============================================================================

# we are Selecting only the independent variables that  we care about for clustering
# Age (mortality driver), Cigarettes_Per_Day (behavioral risk), 
# and BMI (morbidity/physical health risk).
smokers_model_data <- smokers_current %>%
  select(Age, Cigarettes_Per_Day, BMI) %>%
  drop_na() # Just in case there are any missing values

cat("Final modeling dataset size:", nrow(smokers_model_data), "records\n\n")

# Standardization (Z-Score)
# Math: z = (x - mu) / sigma. This ensures Age and BMI don't dominate 
# the Euclidean distance calculation just because their numbers are larger.
smokers_scaled <- scale(smokers_model_data)

# Finding the Optimal Number of Clusters (Silhouette Method)
# we're gonna test k=2 to k=5 to let the math tell us how many distinct risk tiers exist.
library(factoextra)
fviz_nbclust(smokers_scaled, kmeans, method = "silhouette", k.max = 5) +
  labs(title = "Production Model: Finding Optimal Risk Tiers",
       subtitle = "Highest average silhouette width indicates the best k") +
  theme_minimal()

# Running the K-Means Algorithm
# Based on the Silhouette plot that shows 5 as the optimal k but we  
# are setting k to 2 due to it has  roughly close Average Silhouette Width
# so we're gonna take Occam's Razor advice here and build our model on the simplest
#explanation which is 2-tier system rather than a complex 5 
set.seed(1234) # For reproducibility
kmeans_final <- kmeans(smokers_scaled, centers = 2, nstart = 25)

# Validating the 2-Tier Model
library(cluster)
sil_final <- silhouette(kmeans_final$cluster, dist(smokers_scaled))
fviz_silhouette(sil_final) +
  labs(title = "Model Validation: 2-Tier Cluster Homogeneity",
       subtitle = "A score around 0.25 reflects the natural, realistic overlap in human health risk") +
  theme_minimal()

# Interpreting the Business Logic
# The resulting clusters are summarized to define distinct underwriting classes 
# for the pricing and underwriting teams.
cluster_profiles <- smokers_model_data %>%
  mutate(Cluster = as.factor(kmeans_final$cluster)) %>%
  group_by(Cluster) %>%
  summarise(
    Count = n(),
    Avg_Age = round(mean(Age), 1),
    Avg_CPD = round(mean(Cigarettes_Per_Day), 1),
    Avg_BMI = round(mean(BMI), 1)
  ) %>%
  arrange(Avg_CPD) # Sorted by smoking volume to observe the risk gradient

print(cluster_profiles)

# Final Actuarial Visualization
# This scatter plot is generated to visually demonstrate  how 
# the algorithm separates policyholders into two distinct underwriting tiers 
# based on Age and smoking volume.
smokers_model_data %>%
  mutate(Cluster = as.factor(kmeans_final$cluster)) %>%
  ggplot(aes(x = Age, y = Cigarettes_Per_Day, color = Cluster)) +
  geom_point(size = 2, alpha = 0.5) +
  geom_point(
    data = as.data.frame(kmeans_final$centers), 
    aes(x = Age, y = Cigarettes_Per_Day), 
    color = "black", shape = 4, size = 5, stroke = 1.5
  ) +
  labs(
    title = "Final Actuarial Risk Segmentation: Two-Tier Smoker Model",
    subtitle = "Policyholders segmented into Standard and Substandard risk tiers",
    x = "Age (Years)",
    y = "Cigarettes Per Day (CPD)",
    color = "Risk Tier"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14), legend.position = "bottom")


  
