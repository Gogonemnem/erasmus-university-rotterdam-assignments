# install packages in case not installed
# install.packages("dfidx")
# install.packages("mlogit")

# load libraries
library("dfidx")
library("mlogit")

# load data
data <- read.csv(file = "datasets/dataset02.csv")
df <- as.data.frame(data)

# make long dataframe
data_for_cl <- dfidx(data = df[6:ncol(data)],
                     shape = "wide",
                     varying = 1:12,
                     sep = "_",
                     choice = "brandsize_f"
               )

m1 <- mlogit(brandsize_f ~ price + dis + feature, data = data_for_cl)
sum_m1 <- summary(m1)

tempcoeffmat <- sum_m1$CoefTable
cbind(c("Variables", rownames(tempcoeffmat)),
      c("Parameter", tempcoeffmat[, 1]),
      c("Std. Err.", tempcoeffmat[, 2])
)

#log likelihood value
logLik(m1)[1]

#McFadden R-squared
mc_f_r2 <- sum_m1$mfR2[1]


marg1 <- effects(m1, covariate = "price")
