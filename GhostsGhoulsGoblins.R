# GHOSTS, GHOULS, GOBLINS
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(discrim)
library(kernlab)


#### READ IN DATA ####
ggg_test = vroom("./test.csv")
ggg_train = vroom("./train.csv")
ggg_na = vroom("./trainWithMissingValues.csv")

## CLEAN THE DATA
ggg_na$color = factor(ggg_na$color)
ggg_na$type = factor(ggg_na$type)


#### EDA ####
# Start with the least amount of missing values
sum(is.na(ggg_na['bone_length']))   # = 31 NAs
sum(is.na(ggg_na['rotting_flesh']))  # = 22 NAs
sum(is.na(ggg_na['hair_length']))  # = 21 NAs
sum(is.na(ggg_na['has_soul']))  # = 0 NAs
sum(is.na(ggg_na['color']))  # = 0 NAs

#### RECIPE ####
ggg_recipe <- recipe(type~., data=ggg_na) %>% 
  step_impute_knn('hair_length', impute_with=imp_vars('has_soul', 'color'), neighbors=7) %>% 
  step_impute_knn('rotting_flesh', impute_with=imp_vars('hair_length', 'has_soul', 'color'), neighbors=7) %>% 
  step_impute_knn('bone_length', impute_with=imp_vars('rotting_flesh', 'hair_length', 
                                                      'has_soul', 'color'), neighbors=7)

ggg_prepped = prep(ggg_recipe)
baked <- bake(ggg_prepped, new_data = ggg_na)

#### RMSE of IMPUTATIONS ####
rmse_vec(ggg_train[is.na(ggg_na)],
         baked[is.na(ggg_na)])


