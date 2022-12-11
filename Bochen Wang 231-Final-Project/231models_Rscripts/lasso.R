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

phone_lasso_spec <- 
  multinom_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet")

lasso_workflow <- workflow() %>% 
  add_recipe(phone_recipe) %>% 
  add_model(phone_lasso_spec)

penalty_grid <- grid_regular(penalty(range = c(-5, 5)), mixture(range=c(0,1)),
                             levels = 5)
penalty_grid

tune_res_lasso <- tune_grid(
  lasso_workflow,
  resamples = train_folds, 
  grid = penalty_grid
)
autoplot(tune_res_lasso)

best_penalty<-select_best(tune_res_lasso,metrix="roc_auc")
best_penalty

lasso_final<-finalize_workflow(lasso_workflow,best_penalty)

lasso_final_fit <- fit(lasso_final, data = phone_train)

predict(lasso_final_fit,new_data=phone_test,type="class")

test_acc<-augment(lasso_final_fit,new_data=phone_test) %>%
  accuracy(truth=price_range,estimate=.pred_class)
test_acc
