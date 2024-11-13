# GHOSTS, GHOULS, GOBLINS
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(discrim)
library(kernlab)
library(bonsai)
library(lightgbm)


#### READ IN DATA ####
ggg_test = vroom("./test.csv")
ggg_train = vroom("./train.csv")

## CLEAN THE DATA
ggg_train$type = factor(ggg_train$type)


#### EDA ####


#### RECIPE ####
### NAIVE BAYES ####
ggg_recipe <- recipe(type~., data=ggg_train) %>% 
  step_mutate(id, features = id) %>% 
  step_mutate(color = as.factor(color))

ggg_prepped = prep(ggg_recipe)
baked <- bake(ggg_prepped, new_data = ggg_train)

### MODEL ####
nb_model <- naive_Bayes(Laplace=tune(),
                        smoothness=tune()) %>%  
  set_engine("naivebayes") %>%
  set_mode("classification")

## Put into a workflow
nb_workflow <- workflow() %>% 
  add_recipe(ggg_recipe) %>% 
  add_model(nb_model)

## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels=5)


#### CV ####
## Split data for CV
folds <- vfold_cv(ggg_train, v=5, repeats=1)

## Run the CV
tuned_nb <- nb_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc, f_meas, sens, recall,
                               accuracy))
# metric_set(roc_auc, f_meas, sens, recall, spec, precision, accuracy)


## Find best tuning parameters
best_tune <- tuned_nb %>% 
  select_best(metric="roc_auc")

#### Finalize the workflow and fit it ####
final_wf <- nb_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data=ggg_train)

#### Make predictions ####
ggg_preds <- final_wf %>% 
  predict(new_data = ggg_test, type="class")

## Format predictions for kaggle upload
recipe_kaggle_submission <- ggg_preds %>% 
  rename(type=.pred_class) %>% 
  bind_cols(., ggg_test) %>% 
  select(id, type)

## Write out file
vroom_write(x=recipe_kaggle_submission, file="./NBPreds.csv", delim=",")



