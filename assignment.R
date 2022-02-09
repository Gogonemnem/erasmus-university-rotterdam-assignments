# install package in case not installed
# install.packages("mlogit")

# load libraries
library(mlogit)

# load data
data <- read.csv(file = "./datasets/dataset02.csv")
df <- as.data.frame(data)

# make long dataframe 
# could be skipped if you downloaded CL_datasets
data_for_cl <- dfidx(data = df[6:ncol(data)],
                     shape = "wide",
                     varying = 1:12,
                     sep = "_",
                     choice = "brandsize_f"
               )

# , reflevel = "Heinz28" <- automatically done
# fit the CL MN model
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

# partial effects price
marg1 <- effects(m1, covariate = "price")
par_effect_mat_price <- matrix(rep(0, 16), 4, 4)
colnames(par_effect_mat_price) <- colnames(marg1)
rownames(par_effect_mat_price) <- rownames(marg1)

for (j in 1:4) {
    for (i in 1:4) {
        par_effect_mat_price[i, j] <- -mean(m1$coefficients["price"] *
                                      m1$probabilities[, i] *
                                      m1$probabilities[, j]
                                    )
    }
}

for (j in 1:4) {
    par_effect_mat_price[j, j] <- mean(m1$coefficients["price"] *
                                  m1$probabilities[, j] *
                                  (1 - m1$probabilities[, j])
                                )
}

# partial effects display hunts conditional
fake_scenario <- data_for_cl
fake_scenario[idx(fake_scenario, 2) %in% c("Heinz40", "Heinz32", "Heinz28"),
                  c("dis", "feature")] <- 0

fake_scenario1 <- fake_scenario
fake_scenario1[idx(fake_scenario1, 2) == "Hunts32", "dis"] <-  1

hunts_dis_effect <- colMeans(predict(m1, newdata = fake_scenario1) -
                             predict(m1, newdata = fake_scenario))

# Hausman
m2 <- mlogit(brandsize_f ~ price + dis + feature, data = data_for_cl, alt.subset = c("Heinz32", "Heinz28", "Hunts32"))
sum_m2 <- summary(m2)
iia <- hmftest(m1, m2)
iia$statistic
iia$p.value
