
## Marketing Models Chapter 5
install.packages("ggplot2")
install.packages("xtable")
install.packages("haven")
install.packages("sandwich")
install.packages("margins")
install.packages("mclogit")
install.packages("mlogit")
install.packages("gridExtra")
install.packages("dfidx")

#load libraries
library(ggplot2)
# see https://github.com/rstudio/cheatsheets/blob/main/data-visualization-2.1.pdf

library(xtable)


#load packages
library(haven) # for reading the dta file
library(sandwich) # for robust Poisson model standard errors
library(margins) # for computing the average marginal effects
#library(DescTools)
#library(pscl)
library(mclogit)
library(mlogit)

library(gridExtra)

library("dfidx")

##### Load data, Histograms ###################################

#read the data
ch5data <- read.csv(file = "CHAPTER5.csv")
ch5data2 <- ch5data[, c(4:ncol(ch5data))]

#convert to conditional logit format
colnames(ch5data2)  <- c("PRIVATE",
                            "SUNSHINE",
                            "KEEBLER",
                            "NABISCO",
                            "PRICE_PRIVATE",
                            "PRICE_SUNSHINE",
                            "PRICE_KEEBLER",
                            "PRICE_NABISCO",
                            "DISPL_PRIVATE",
                            "DISPL_SUNSHINE",
                            "DISPL_KEEBLER",
                            "DISPL_NABISCO",
                            "FEAT_PRIVATE",
                            "FEAT_SUNSHINE",
                            "FEAT_KEEBLER",
                            "FEAT_NABISCO",
                            "FEATDISPL_PRIVATE",
                            "FEATDISPL_SUNSHINE",
                            "FEATDISPL_KEEBLER",
                            "FEATDISPL_NABISCO"#,
                            #"OBS",
                            #"HOUSEHOLDID",
                            #"LASTPURCHASE"
                         )

choices_binary <- as.matrix(ch5data2[, c(1:4)])

ch5df <- as.data.frame(ch5data2)
ch5df$brand_f <-  as.factor(colnames(choices_binary)[choices_binary %*% 1:ncol(choices_binary)])
ch5df$obs_id <- 1:nrow(ch5df)


ch5data_for_CL <- dfidx(data = ch5df[5:ncol(ch5df)], 
                       shape = "wide", 
                       varying = 1:16, 
                       sep = "_",  
                       choice = "brand_f" )



# counts (or sums of weights)
g <- ggplot(ch5df, aes(brand_f))
# Number of cars in each class:
g + geom_bar() + xlab("Saltine Cracker Brand")




#Slide 7 data characteristics table

datachar <- rbind(
(100*table(ch5df$brand_f)/nrow(ch5df))[c("PRIVATE", "SUNSHINE", "KEEBLER",  "NABISCO")],
c(mean(ch5df$PRICE_PRIVATE) , 
  mean(ch5df$PRICE_SUNSHINE) ,
  mean(ch5df$PRICE_KEEBLER) ,
  mean(ch5df$PRICE_NABISCO)),
c(100*mean(ch5df$DISPL_PRIVATE) , 
  100*mean(ch5df$DISPL_SUNSHINE) ,
  100*mean(ch5df$DISPL_KEEBLER) ,
  100*mean(ch5df$DISPL_NABISCO)),
c(100*mean(ch5df$FEAT_PRIVATE) , 
  100*mean(ch5df$FEAT_SUNSHINE) ,
  100*mean(ch5df$FEAT_KEEBLER) ,
  100*mean(ch5df$FEAT_NABISCO)),
c(100*mean(ch5df$FEATDISPL_PRIVATE) , 
  100*mean(ch5df$FEATDISPL_SUNSHINE) ,
  100*mean(ch5df$FEATDISPL_KEEBLER) ,
  100*mean(ch5df$FEATDISPL_NABISCO))
)

rownames(datachar) <- c("Choice percentage", 
                        "Average price (US$)",
                        "% display only",
                        "% feature only",
                        "% feature and display only")

#if want to save table as TeX file
#use xtable
#add footnotes in TeX editor
# textemp <- xtable(datachar, caption = "Characteristics of the dependent variable and explanatory variables: the choice between four brands of saltine crackers")
# print(textemp, file = "FILE NAME")


#///////////////////////////////////////////////////////////////////////////////

##### #Multinomial Logit Probabilities Example ####################################################################

#Slide 14

#x-values
xtemp <- seq(from = -5, to = 5, by = 0.05)

b01 <- -1
b02 <- 1
b03 <- 0
b11 <- 1
b12 <- 0.5
b13 <- 0


pY1 <- exp(b01 + b11*xtemp)/( 1 + exp(b01 + b11*xtemp) + exp(b02 + b12*xtemp) )
pY2 <- exp(b02 + b12*xtemp)/( 1 + exp(b01 + b11*xtemp) + exp(b02 + b12*xtemp) )
pY3 <- 1/( 1 + exp(b01 + b11*xtemp) + exp(b02 + b12*xtemp) )

MNLdataall <- data.frame(y = c(pY1,pY2,pY3) ,
                           x1= c(xtemp, xtemp, xtemp),
                           params = factor(c(rep("a", length(xtemp)), 
                                             rep("b", length(xtemp)), 
                                             rep("c", length(xtemp)))))

ggplot(data = MNLdataall, aes(y = y, x= x1, group = params, shape = params, color = params)) +
  geom_point()+geom_line()+
  xlim(-5, 5)  +
  labs(x = "x_i", 
       y = "Pr[Y_i = j]",
       title= "Scatter Diagram of logistic function of b0+b1*x_i against x_i")+
  scale_color_discrete(name  ="Outcomes and Parameters",
                       breaks=c("a", "b", "c"),
                       labels=c("j = 1, b01=-1, b11=1", "j=2, b02=1, b12=0.5", "j = 3, b03=0, b13=0"))+
  scale_shape_discrete(name  ="Outcomes and Parameters",
                       breaks=c("a", "b", "c"),
                       labels=c("j = 1, b01=-1, b11=1", "j=2, b02=1, b12=0.5", "j = 3, b03=0, b13=0"))




#///////////////////////////////////////////////////////////////////////////////

##### #Conditional Logit Probabilities Example ####################################################################

#Slide 14

#x-values
xtemp <- seq(from= -5, to = 5, by = 0.05)

b01 <- 1
b02 <- -1
b03 <- 0
gamma1 <- -2


#xtemp is w_{i1}
#ASSUMING w_{i3} and w_{i3} both equal 1 for all i.

pY1 <- exp(b01 + gamma1*xtemp)/( exp(b01 + gamma1*xtemp) + exp(b02 + gamma1*1) + exp(b03 + gamma1*1) )
pY2 <- exp(b02 + gamma1*1)/( exp(b01 + gamma1*xtemp) + exp(b02 + gamma1*1) + exp(b03 + gamma1*1) )
pY3 <- exp(b03 + gamma1*1)/( exp(b01 + gamma1*xtemp) + exp(b02 + gamma1*1) + exp(b03 + gamma1*1) )

MNLdataall <- data.frame(y = c(pY1,pY2,pY3) ,
                         x1= c(xtemp, xtemp, xtemp),
                         params = factor(c(rep("a", length(xtemp)), 
                                           rep("b", length(xtemp)), 
                                           rep("c", length(xtemp)))))

ggplot(data = MNLdataall, aes(y = y, x= x1, group = params, shape = params, color = params)) +
  geom_point()+geom_line()+
  xlim(-5, 5)  +
  labs(x = "x_i", 
       y = "Pr[Y_i = j]",
       title= "Scatter Diagram of logistic function of b0+b1*x_i against x_i")+
  scale_color_discrete(name  ="Outcomes and Parameters",
                       breaks=c("a", "b", "c"),
                       labels=c("j = 1, b01=1, gamma1= -2", "j=2, b02=-1, gamma1= -2", "j = 3, b03=0, gamma1= -2"))+
  scale_shape_discrete(name  ="Outcomes and Parameters",
                       breaks=c("a", "b", "c"),
                       labels=c("j = 1, b01=1, gamma1= -2", "j=2, b02=-1, gamma1= -2", "j = 3, b03=0, gamma1= -2"))




#///////////////////////////////////////////////////////////////////////////////

##### Conditional Logit Model for Probability of Purchasing Saltine Cracker Brand ####################################################################

library(mlogit)
#read the data
ch5data <- read.csv(file = "CHAPTER5.csv")

training_data <- ch5data[ch5data$LASTPURCHASE == 0, ]
test_data <- ch5data[ch5data$LASTPURCHASE == 1, ]

training_data <- training_data[, c(4:ncol(training_data))]
test_data <- test_data[, c(4:ncol(test_data))]

#convert to conditional logit format
colnames(training_data)  <- c("PRIVATE",
                         "SUNSHINE",
                         "KEEBLER",
                         "NABISCO",
                         "PRICE_PRIVATE",
                         "PRICE_SUNSHINE",
                         "PRICE_KEEBLER",
                         "PRICE_NABISCO",
                         "DISPL_PRIVATE",
                         "DISPL_SUNSHINE",
                         "DISPL_KEEBLER",
                         "DISPL_NABISCO",
                         "FEAT_PRIVATE",
                         "FEAT_SUNSHINE",
                         "FEAT_KEEBLER",
                         "FEAT_NABISCO",
                         "FEATDISPL_PRIVATE",
                         "FEATDISPL_SUNSHINE",
                         "FEATDISPL_KEEBLER",
                         "FEATDISPL_NABISCO"#,
                         #"OBS",
                         #"HOUSEHOLDID",
                         #"LASTPURCHASE"
)

choices_binary <- as.matrix(training_data[, c(1:4)])

training_datadf <- as.data.frame(training_data)
training_datadf$brand_f <-  as.factor(colnames(choices_binary)[choices_binary %*% 1:ncol(choices_binary)])
training_datadf$obs_id <- 1:nrow(training_datadf)

training_data_for_CL <- dfidx(data = training_datadf[5:ncol(training_datadf)],
                        shape = "wide",
                        varying = 1:16,
                        sep = "_",
                        choice = "brand_f")


colnames(test_data)  <- c("PRIVATE",
                              "SUNSHINE",
                              "KEEBLER",
                              "NABISCO",
                              "PRICE_PRIVATE",
                              "PRICE_SUNSHINE",
                              "PRICE_KEEBLER",
                              "PRICE_NABISCO",
                              "DISPL_PRIVATE",
                              "DISPL_SUNSHINE",
                              "DISPL_KEEBLER",
                              "DISPL_NABISCO",
                              "FEAT_PRIVATE",
                              "FEAT_SUNSHINE",
                              "FEAT_KEEBLER",
                              "FEAT_NABISCO",
                              "FEATDISPL_PRIVATE",
                              "FEATDISPL_SUNSHINE",
                              "FEATDISPL_KEEBLER",
                              "FEATDISPL_NABISCO"#,
                              #"OBS",
                              #"HOUSEHOLDID",
                              #"LASTPURCHASE"
)

choices_binary <- as.matrix(test_data[, c(1:4)])

test_datadf <- as.data.frame(test_data)
test_datadf$brand_f <-  as.factor(colnames(choices_binary)[choices_binary %*% 1:ncol(choices_binary)])
test_datadf$obs_id <- 1:nrow(test_datadf)

test_data_for_CL <- dfidx(data = test_datadf[5:ncol(test_datadf)],
                              shape = "wide",
                              varying = 1:16,
                              sep = "_",
                              choice = "brand_f")


summary(m1 <-
          mlogit(
            brand_f ~ PRICE  + DISPL  + FEAT + FEATDISPL,
            data = training_data_for_CL
          )
)

sum_m1 <- summary(m1)

tempcoeffmat <- sum_m1$CoefTable


cbind(c("Variables", rownames(tempcoeffmat)),
      c("Parameter", tempcoeffmat[, 1]),
      c("Std. Err.", tempcoeffmat[, 2])
      )







################ Log Likelihoods #############################################


#log likelihood value
logLik(m1)[1]

#McFadden R-squared
McFR2_2 <- sum_m1$mfR2[1]


################ Prediction-Realization Table ##################################


training_predprobs <- predict(m1,training_data_for_CL )
training_predcats <- apply(training_predprobs,1,which.max)

test_predprobs <- predict(m1,test_data_for_CL )
test_predcats <- apply(test_predprobs,1,which.max)


pred_realize_training <- table(as.factor(training_datadf$brand_f),factor(training_predcats, levels = c(1,2,3,4)))/nrow(training_datadf)

colnames(pred_realize_training) <- c("KEEBLER",
                                     "NABISCO",
                                     "PRIVATE",
                                     "SUNSHINE")

hitrate_training <- pred_realize_training[1,1]+
  pred_realize_training[2,2]+
  pred_realize_training[3,3]+
  pred_realize_training[4,4]



pred_realize_test <- table(as.factor(test_datadf$brand_f),factor(test_predcats, levels = c(1,2,3,4)))/nrow(test_datadf)

colnames(pred_realize_test) <- c("KEEBLER",
                                     "NABISCO",
                                     "PRIVATE",
                                     "SUNSHINE")

hitrate_test <- pred_realize_test[1,1]+
  pred_realize_test[2,2]+
  pred_realize_test[3,3]+
  pred_realize_test[4,4]




################ Graphs of Choice Probabilities against Price #############################################

#To graph the choice probabilities from slide 42, can create new datasets and apply predict function

#more straightforward to directly apply formulae





#x-values
xtemp <- seq(from= 0, to = 1.5, by = 0.05)

#nabisco
b01 <-  tempcoeffmat[1]
#private
b02 <-  tempcoeffmat[2]
#sunshine
b03 <-  tempcoeffmat[3]
#keebler
b04 <-  0

gamma1 <- tempcoeffmat[4]

  #assuming no promotions

#xtemp is w_{i1}
#ASSUMING w_{i3} and w_{i3} both equal 1 for all i.


#setting prices of everything other than nabisco to average
#nabisco average price is 1.08

pY1 <- exp(b01 + gamma1*xtemp)/( exp(b01 + gamma1*xtemp) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*1.13))
pY2 <- exp(b02 + gamma1*0.68)/( exp(b01 + gamma1*xtemp) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*1.13))
pY3 <- exp(b03 + gamma1*0.96)/( exp(b01 + gamma1*xtemp) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*1.13))
pY4 <- exp(b04 + gamma1*1.13)/( exp(b01 + gamma1*xtemp) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*1.13))

CLdataall <- data.frame(y = c(pY1,pY2,pY3,pY4) ,
                         x1= c(xtemp, xtemp, xtemp, xtemp),
                         params = factor(c(rep("a", length(xtemp)), 
                                           rep("b", length(xtemp)), 
                                           rep("c", length(xtemp)),
                                           rep("d", length(xtemp)))))

plot_nabisco <- ggplot(data = CLdataall, aes(y = y, x= x1, group = params, shape = params, color = params)) +
  geom_point()+geom_line()+
  xlim(0, 1.5)  +  ylim(0,1) +
  labs(x = "Price Nabisco", 
       y = "Pr[Y_i = j]",
       title= "Purchase Probabilities against Price of Nabisco")+
  scale_color_discrete(name  ="Brand",
                       breaks=c("a", "b", "c", "d"),
                       labels=c("Nabisco", "Private Label", "Sunshine", "Keebler"))+
  scale_shape_discrete(name  ="Brand",
                       breaks=c("a", "b", "c", "d"),
                       labels=c("Nabisco", "Private Label", "Sunshine", "Keebler"))


#similar graph for private label


pY1 <- exp(b01 + gamma1*1.08)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*xtemp) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*1.13))
pY2 <- exp(b02 + gamma1*xtemp)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*xtemp) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*1.13))
pY3 <- exp(b03 + gamma1*0.96)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*xtemp) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*1.13))
pY4 <- exp(b04 + gamma1*1.13)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*xtemp) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*1.13))

CLdataall <- data.frame(y = c(pY1,pY2,pY3,pY4) ,
                        x1= c(xtemp, xtemp, xtemp, xtemp),
                        params = factor(c(rep("a", length(xtemp)), 
                                          rep("b", length(xtemp)), 
                                          rep("c", length(xtemp)),
                                          rep("d", length(xtemp)))))

plot_private <- ggplot(data = CLdataall, aes(y = y, x= x1, group = params, shape = params, color = params)) +
  geom_point()+geom_line()+
  xlim(0, 1.5)  + ylim(0,1) +
  labs(x = "Price Private Label", 
       y = "Pr[Y_i = j]",
       title= "Purchase Probabiltiies against Price of Private Label")+
  scale_color_discrete(name  ="Brand",
                       breaks=c("a", "b", "c", "d"),
                       labels=c("Nabisco", "Private Label", "Sunshine", "Keebler"))+
  scale_shape_discrete(name  ="Brand",
                       breaks=c("a", "b", "c", "d"),
                       labels=c("Nabisco", "Private Label", "Sunshine", "Keebler"))



  
  #similar graph for sunshine
  
  
  pY1 <- exp(b01 + gamma1*1.08)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*xtemp) + exp(b04 + gamma1*1.13))
  pY2 <- exp(b02 + gamma1*0.68)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*xtemp) + exp(b04 + gamma1*1.13))
  pY3 <- exp(b03 + gamma1*xtemp)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*xtemp) + exp(b04 + gamma1*1.13))
  pY4 <- exp(b04 + gamma1*1.13)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*xtemp) + exp(b04 + gamma1*1.13))
  
  CLdataall <- data.frame(y = c(pY1,pY2,pY3,pY4) ,
                          x1= c(xtemp, xtemp, xtemp, xtemp),
                          params = factor(c(rep("a", length(xtemp)), 
                                            rep("b", length(xtemp)), 
                                            rep("c", length(xtemp)),
                                            rep("d", length(xtemp)))))
  
  plot_sunshine <-   ggplot(data = CLdataall, aes(y = y, x= x1, group = params, shape = params, color = params)) +
    geom_point()+geom_line()+
    xlim(0, 1.5)  + ylim(0,1) +
  labs(x = "Price Sunshine", 
       y = "Pr[Y_i = j]",
       title= "Purchase Probabiltiies against Price of Private Label")+
    scale_color_discrete(name  ="Brand",
                         breaks=c("a", "b", "c", "d"),
                         labels=c("Nabisco", "Private Label", "Sunshine", "Keebler"))+
    scale_shape_discrete(name  ="Brand",
                         breaks=c("a", "b", "c", "d"),
                         labels=c("Nabisco", "Private Label", "Sunshine", "Keebler"))
  
  
  
  
  
  
  #similar graph for Keebler
  
  
  pY1 <- exp(b01 + gamma1*1.08)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*xtemp))
  pY2 <- exp(b02 + gamma1*0.68)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*xtemp))
  pY3 <- exp(b03 + gamma1*0.96)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*xtemp))
  pY4 <- exp(b04 + gamma1*xtemp)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*xtemp))
  
  CLdataall <- data.frame(y = c(pY1,pY2,pY3,pY4) ,
                          x1= c(xtemp, xtemp, xtemp, xtemp),
                          params = factor(c(rep("a", length(xtemp)), 
                                            rep("b", length(xtemp)), 
                                            rep("c", length(xtemp)),
                                            rep("d", length(xtemp)))))
  
  plot_keebler <- ggplot(data = CLdataall, aes(y = y, x= x1, group = params, shape = params, color = params)) +
    geom_point()+geom_line()+
    xlim(0, 1.5)  + ylim(0,1) +
  labs(x = "Price Keebler", 
       y = "Pr[Y_i = j]",
       title= "Purchase Probabiltiies against Price of Private Label")+
    scale_color_discrete(name  ="Brand",
                         breaks=c("a", "b", "c", "d"),
                         labels=c("Nabisco", "Private Label", "Sunshine", "Keebler"))+
    scale_shape_discrete(name  ="Brand",
                         breaks=c("a", "b", "c", "d"),
                         labels=c("Nabisco", "Private Label", "Sunshine", "Keebler"))
  
  
  
  
  grid.arrange(plot_private, plot_sunshine, plot_keebler , plot_nabisco, ncol=2, nrow = 2)
  
  
  
################ Graphs of Elasticities against Price #############################################

  #Own-price elasticities, setting other prices to sample mean
  
  pY1 <- exp(b01 + gamma1*xtemp)/( exp(b01 + gamma1*xtemp) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*1.13))
  pY2 <- exp(b02 + gamma1*xtemp)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*xtemp) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*1.13))
  pY3 <- exp(b03 + gamma1*xtemp)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*xtemp) + exp(b04 + gamma1*1.13))
  pY4 <- exp(b04 + gamma1*xtemp)/( exp(b01 + gamma1*1.08) + exp(b02 + gamma1*0.68) + exp(b03 + gamma1*0.96) + exp(b04 + gamma1*xtemp))
  
  elas_1 <- gamma1*xtemp*pY1*(1-pY1)
  elas_2 <- gamma1*xtemp*pY2*(1-pY2)
  elas_3 <- gamma1*xtemp*pY3*(1-pY3)
  elas_4 <- gamma1*xtemp*pY4*(1-pY4)
  
  
  
  elasdataall <- data.frame(y = c(elas_1,elas_2,elas_3,elas_4) ,
                          x1= c(xtemp, xtemp, xtemp, xtemp),
                          params = factor(c(rep("a", length(xtemp)), 
                                            rep("b", length(xtemp)), 
                                            rep("c", length(xtemp)),
                                            rep("d", length(xtemp)))))
  
  plot_elasticities <- ggplot(data = elasdataall, aes(y = y, x= x1, group = params, shape = params, color = params)) +
    geom_point()+geom_line()+
    xlim(0, 1.5)  + ylim(-1,0) +
    labs(x = "Price", 
         y = "Pr[Y_i = j ]",
         title= "Purchase Probabiltiies against Price")+
    scale_color_discrete(name  ="Brand (j)",
                         breaks=c("a", "b", "c", "d"),
                         labels=c("Nabisco", "Private Label", "Sunshine", "Keebler"))+
    scale_shape_discrete(name  ="Brand (j)",
                         breaks=c("a", "b", "c", "d"),
                         labels=c("Nabisco", "Private Label", "Sunshine", "Keebler"))
  
  
  
  
  ################# average partial effect of price #####################################

  training_data_for_CL
  #partial effect at the average
  marg1 <- effects(m1, covariate = "PRICE")
  
  
  par_effect_mat_price0 <- matrix(rep(0,16),4,4)
  colnames(par_effect_mat_price0) <- colnames(marg1)
  rownames(par_effect_mat_price0) <- rownames(marg1)
  
  print(length(unique(training_data_for_CL$idx$id1)))
  for(i in 1:length(unique(training_data_for_CL$idx$id1))){
    print(i)
    z <- data.frame(PRICE = training_data_for_CL[ training_data_for_CL$idx$id1 == i , "PRICE" ],
                    DISPL = training_data_for_CL[ training_data_for_CL$idx$id1 == i , "DISPL" ],
                    FEAT = training_data_for_CL[ training_data_for_CL$idx$id1 == i , "FEAT" ],
                    FEATDISPL = training_data_for_CL[ training_data_for_CL$idx$id1 == i , "FEATDISPL" ])
    
    par_effect_mat_price0 <- par_effect_mat_price0 + effects(m1, covariate = "PRICE", data = z)
    
    
  } 
  
  par_effect_mat_price0 <- par_effect_mat_price0/length(unique(training_data_for_CL$idx$id1))
  
  
  
  
  par_effect_mat_price <- matrix(rep(0,16),4,4)
  colnames(par_effect_mat_price) <- colnames(marg1)
  rownames(par_effect_mat_price) <- rownames(marg1)
  
  
  for(j in 1:4){
    for(i in 1:4){
      par_effect_mat_price[i,j] <- -mean(m1$coefficients["PRICE"]*
                                           m1$probabilities[,i]*(m1$probabilities[,j]))
    }
  }
  
  for(j in 1:4){
    par_effect_mat_price[j,j] <- mean(m1$coefficients["PRICE"]*
                                        m1$probabilities[,j]*(1-m1$probabilities[,j]))
  }
  

  ################# average effect of display promotion #####################################
  
idx(training_data_for_CL_new0, 2) %in% c("KEEBLER", "NABISCO", "PRIVATE")
  
  training_data_for_CL_new0 <- training_data_for_CL
  training_data_for_CL_new0[idx(training_data_for_CL_new0, 2) == "KEEBLER", c("DISPL", "FEAT", "FEATDISPL")] <- 0
  training_data_for_CL_new0[idx(training_data_for_CL_new0, 2) == "KEEBLER", "DISPL"] <-  0
  training_data_for_CL_new0[idx(training_data_for_CL_new0, 2) == "KEEBLER", "FEAT"] <-  0
  training_data_for_CL_new0[idx(training_data_for_CL_new0, 2) == "KEEBLER", "FEATDISPL"] <-  0
  
  training_data_for_CL_new0[idx(training_data_for_CL_new0, 2) == "NABISCO", "DISPL"] <-  0
  training_data_for_CL_new0[idx(training_data_for_CL_new0, 2) == "NABISCO", "FEAT"] <-  0
  training_data_for_CL_new0[idx(training_data_for_CL_new0, 2) == "NABISCO", "FEATDISPL"] <-  0
  
  training_data_for_CL_new0[idx(training_data_for_CL_new0, 2) == "PRIVATE", "DISPL"] <-  0
  training_data_for_CL_new0[idx(training_data_for_CL_new0, 2) == "PRIVATE", "FEAT"] <-  0
  training_data_for_CL_new0[idx(training_data_for_CL_new0, 2) == "PRIVATE", "FEATDISPL"] <-  0
  
  training_data_for_CL_new0[idx(training_data_for_CL_new0, 2) == "SUNSHINE", "DISPL"] <-  0
  training_data_for_CL_new0[idx(training_data_for_CL_new0, 2) == "SUNSHINE", "FEAT"] <-  0
  training_data_for_CL_new0[idx(training_data_for_CL_new0, 2) == "SUNSHINE", "FEATDISPL"] <-  0
  
  
  training_data_for_CL_new1 <- training_data_for_CL_new0
  
  training_data_for_CL_new1[idx(training_data_for_CL_new1, 2) == "SUNSHINE", "DISPL"] <-  1
  
  
  training_data_for_CL_new2 <- training_data_for_CL_new0
  
  training_data_for_CL_new2[idx(training_data_for_CL_new2, 2) == "SUNSHINE", "FEAT"] <-  1
  
  
  
  SUNSHINEdis_effect <- colMeans(predict(m1, newdata = training_data_for_CL_new1) - predict(m1, newdata = training_data_for_CL_new0))
  
  SUNSHINEfeature_effect <- colMeans(predict(m1, newdata = training_data_for_CL_new2) - predict(m1, newdata = training_data_for_CL_new0))
  
  SUNSHINE_effects_mat <- cbind(c("display effect", "feature effect"),
                             rbind(SUNSHINEdis_effect, SUNSHINEfeature_effect))
