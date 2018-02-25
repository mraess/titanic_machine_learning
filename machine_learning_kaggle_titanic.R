
# Titanic - prediction comparisons ----------------------------------------

# Reading in data ---------------------------------------------------------

library(tidyverse)

training <- read.csv("trainTi.csv", header = TRUE)

testing <- read.csv("testTi.csv", header = TRUE)

dim(training); dim(testing)

str(training); summary(training)

head(training)

## Deciding on variables to include in model - EPA

table(training[ ,c("Survived", "Pclass")]) # Looks like a good predictor of "Survived"

table(training[ ,c("Survived", "Sex")]) # No NAs - seems to be a good predictor - include

table(training[ ,c("Survived", "Age")]) # 177 NAs - exclude for now

ggplot(training, aes(Survived, Age)) + geom_bar(stat = "identity")

table(training[ ,c("Survived", "Embarked")]) # include

ggplot(training, aes(Survived, Parch)) + geom_bar(stat = "identity") # include

ggplot(training, aes(Survived, SibSp)) + geom_bar(stat = "identity") # include

summary(training$Fare)

ggplot(training, aes(as.factor(Survived), Fare)) + geom_boxplot() # No NAs - boxplots are different - include


# Convert "Survive" to factor ----------------------------------------------------------------

training$Survived <- factor(training$Survived, levels = c("0", "1"), labels = c("0", "1"))

training$Survived <- as.factor(ifelse(training$Survived == "0", "died", "survived"))

summary(training$Survived)

# Test random forrest model -----------------------------------------------

library(caret)

myControl <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE)

model_rf <- train(Survived ~ Pclass + Sex + Fare + SibSp + Embarked + Parch, data = training, method = "ranger",
                  trControl = myControl)

model_rf


testing$Survived <- predict(model_rf, testing) ## error - different lengths

summary(testing) # NA in Fare

## Use median to impute missing value

testing$Fare <- ifelse(is.na(testing$Fare), median(testing$Fare, na.rm = TRUE), testing$Fare)

## Try again

testing$Survived <- predict(model_rf, testing)

# Test accuracy with training data since Kaggle does the actual eval on test set

training$Survived_pred <- predict(model_rf, training)

confusionMatrix(training$Survived, training$Survived_pred)

## Prepare for Kaggle submission

testing$Survived <- as.factor(ifelse(testing$Survived == "died", "0", "1"))

submission <- testing %>% select(PassengerId, Survived)

write.table(submission, file = "submission.csv", col.names = TRUE, row.names = FALSE, sep = ",")


## .7655 accuracy - room for improvement


## Include age and impute missing values with median

training$Age <- ifelse(is.na(training$Age), median(training$Age, na.rm = TRUE), training$Age)

summary(training)

model_rf2 <- train(Survived ~ Pclass + Sex + Fare + SibSp + Embarked + Parch + Age, data = training, method = "ranger",
                   trControl = myControl)

model_rf2

testing2 <- testing

## Use median to impute missing value

testing2$Fare <- ifelse(is.na(testing2$Fare), median(testing2$Fare, na.rm = TRUE), testing2$Fare)

testing2$Age <- ifelse(is.na(testing2$Age), median(testing2$Age, na.rm = TRUE), testing2$Age)

testing2$Survived <- predict(model_rf2, testing2)

# New submission to Kaggle

testing2$Survived <- as.factor(ifelse(testing2$Survived == "died", "0", "1"))

submission2 <- testing2 %>% select(PassengerId, Survived)

write.table(submission2, file = "submission2.csv", col.names = TRUE, row.names = FALSE, sep = ",")


### Improvement to .77990 and an advancement of 2,426 places on Kaggle leaderboard


# Creating a “Deck” variable from “Cabin" ---------------------------------

training$Deck <- as.factor(substring(training$Cabin, 1, 1)) # removes actual numbers and reduces it to cabin allocations

summary(training$Deck) # 687 are not assigned

# Randomly assign decks based on Pclass
# Based on the data from encyclopedia titanica the letters give the deck A(top) - G(bottom)

noDeck <- which(training$Deck == "")

training$Deck[noDeck] = NA; sum(is.na(training$Deck))

## Assign deck

set.seed(666)

training$Deck[is.na(training$Deck)] <- ifelse(training$Pclass == "First", sample(c("A", "B", "C", "D", "E"), replace = TRUE),
                                              ifelse(training$Pclass == "Second", sample(c("D", "E", "F"), replace = TRUE), 
                                                     sample(c("E", "F", "G"), replace = TRUE)))


### 

table(training$Pclass, training$Deck) # might not be a good predictor since there is some overlap
# in decks D through G in terms of classes (mixed)


model_rf3 <- train(Survived ~ Pclass + Sex + Fare + SibSp + Embarked + Parch + Age + Deck, data = training, method = "ranger",
                   trControl = myControl)

model_rf3

testing3 <- testing

## Use median to impute missing value

testing3$Fare <- ifelse(is.na(testing3$Fare), median(testing3$Fare, na.rm = TRUE), testing3$Fare)

testing3$Age <- ifelse(is.na(testing3$Age), median(testing3$Age, na.rm = TRUE), testing3$Age)

testing3$Deck <- as.factor(substring(testing3$Cabin, 1, 1))

testing3$Deck[is.na(testing3$Deck)] <- ifelse(training$Pclass == "First", sample(c("A", "B", "C", "D", "E"), replace = TRUE),
                                              ifelse(training$Pclass == "Second", sample(c("D", "E", "F"), replace = TRUE), 
                                                     sample(c("E", "F", "G"), replace = TRUE)))

testing3$Survived <- predict(model_rf3, testing3)

# Third submission to Kaggle

testing3$Survived <- as.factor(ifelse(testing3$Survived == "died", "0", "1"))

submission3 <- testing3 %>% select(PassengerId, Survived)

write.table(submission3, file = "submission3.csv", col.names = TRUE, row.names = FALSE, sep = ",")

### .76076 - inclusion of the Deck variable made the model perform worse

## Try different model with same variables as in model_rf2

