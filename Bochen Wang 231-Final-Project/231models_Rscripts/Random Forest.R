library(corrplot)  # for the correlation plot
library(discrim)  # for linear discriminant analysis
library(corrr)   # for calculating correlation
library(knitr)   # to help with the knitting process
library(MASS)    # to assist with the markdown processes
library(tidyverse)   # using tidyverse and tidymodels for this project mostly
library(tidymodels)
library(ggplot2)   # for most of our visualizations
library(ggrepel)
library(klaR)
library(rpart.plot)  # for visualizing trees
library(vip)         # for variable importance 
library(janitor)     # for cleaning out our data
library(randomForest)   # for building our randomForest
library(stringr)    # for matching strings
library(poissonreg)

library("dplyr")     # for basic r functions
library("yardstick") # for measuring certain metrics
tidymodels_prefer()
options(scipen = 9999)

phone <- read.csv("D:/1/code/pstat131/train.csv")

phone$blue <- as.factor(phone$blue)
phone$dual_sim <- as.factor(phone$dual_sim)
phone$four_g <- as.factor(phone$four_g)
phone$three_g <- as.factor(phone$three_g)
phone$touch_screen <- as.factor(phone$touch_screen)
phone$wifi <- as.factor(phone$wifi)
phone$price_range <- as.factor(phone$price_range)

phone <- phone %>% 
  mutate(px_size = px_width * px_height)

set.seed(2500)  # setting a seed for reproducible results
phone_split <- phone %>%
  initial_split(prop = 0.8, strata = price_range)

phone_train <- training(phone_split) # training split
phone_test <- testing(phone_split) # testing split


train_folds <- vfold_cv(phone_train, v = 5, strata = price_range)  # 5-fold CV

#phone_recipe <-   # building the recipe to be used for each model
#  recipe(price_range ~ battery_power + blue + clock_speed + dual_sim +
#           four_g + fc + pc + int_memory + m_dep + mobile_wt + n_cores +
#           ram + talk_time + three_g + touch_screen + wifi , 
#         data = phone_train) %>% 
#  step_dummy(all_nominal_predictors()) %>% 
#  step_interact(terms=~sc_w:sc_h) %>% 
#  step_interact(terms=~px_width:px_height) %>% 
#  step_interact(terms=~fc:pc) %>% 
#  step_center(all_predictors()) %>%   # standardizing predictors
#  step_scale(all_predictors())

phone_recipe <-   # building the recipe to be used for each model
  recipe(price_range ~ battery_power + blue + clock_speed + dual_sim +
           four_g + fc + pc + int_memory + m_dep + mobile_wt + n_cores +
           ram + talk_time + three_g + touch_screen + wifi + px_size, 
         data = phone) %>% 
  step_impute_linear(clock_speed) %>% 
  step_impute_linear(m_dep) %>% 
  #still need to set interactions here!!!!!!!!!!!!!
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(terms=~fc:pc) %>% 
  step_center(all_predictors()) %>%   # standardizing predictors
  step_scale(all_predictors())

library(ranger)
rand_for <- rand_forest() %>%
  set_engine("ranger", importance = "impurity") 
rand_for_spec <- rand_for %>%
  set_mode("classification")
rand_for_wf <- workflow() %>%
  add_model(rand_for_spec %>% set_args(mtry = tune())
            %>% set_args(trees = tune()) %>% 
              set_args(min_n = tune())) %>% 
  add_recipe(phone_recipe)
param_grid_rand <- grid_regular(mtry(range = c(1, 18)),trees(range = c(1,10)),
                             min_n(range = c(1, 10)),levels = 5)

tune_res_rand <- tune_grid(
  rand_for_wf, 
  resamples = train_folds, 
  grid = param_grid_rand, 
  metrics = metric_set(roc_auc)
)
autoplot(tune_res_rand)

best_rand<-select_best(tune_res_rand,metrix="roc_auc")
best_rand

rand_final<-finalize_workflow(rand_for_wf, best_rand)

rand_final_fit <- fit(rand_final, data = phone_train)

predict(rand_final_fit,new_data=phone_test,type="class")

test_acc<-augment(rand_final_fit,new_data=phone_test) %>%
  accuracy(truth=price_range,estimate=.pred_class)
test_acc
