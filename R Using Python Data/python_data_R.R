# takes in python data and runs NNGP
library(spNNGP)

main = ".."

# files
  # model
file.x = main + "/nngp-downscaling/R Using Python Data/Data From Python/x_train.csv"
file.y = main + "/nngp-downscaling/R Using Python Data/Data From Python/y_train.csv"
file.coords = main + "/nngp-downscaling/R Using Python Data/Data From Python/coords_train.csv"

  # predict
file.x.ho = main + "/nngp-downscaling/R Using Python Data/Data From Python/x_test.csv"
file.coords.ho = main + "/nngp-downscaling/R Using Python Data/Data From Python/coords_test.csv"
file.y.ho = main + "/nngp-downscaling/R Using Python Data/Data From R/y_hat.csv"

# variable from python
  # model
x = as.matrix(read.csv(file.x))[, 2:7]  # 2:7 outside number changed for how many x's
y = as.matrix(read.csv(file.y))[, 2]
coords = as.matrix(read.csv(file.coords))[, 2:3]
  # predict
x.ho = as.matrix(read.csv(file.x.ho))[, 2:7]
coords.ho = as.matrix(read.csv(file.coords.ho))[, 2:3]

# variables needed

sigma.sq <- 5
tau.sq <- 1
phi <- 3/0.5

##Fit a Response, Latent, and Conjugate NNGP model
n.samples <- 5

starting <- list("phi"=phi, "sigma.sq"=5, "tau.sq"=1)
tuning <- list("phi"=0.5, "sigma.sq"=0.5, "tau.sq"=0.5)
priors <- list("phi.Unif"=c(3/1, 3/0.01), "sigma.sq.IG"=c(2, 5), "tau.sq.IG"=c(2, 1))

cov.model <- "exponential"
n.report <- 5

############
# Response #
############

sim.r <- spNNGP(y~x-1, coords=coords, starting=starting, method="response", n.neighbors=10,
                tuning=tuning, priors=priors, cov.model=cov.model,
                n.samples=n.samples, n.omp.threads=1, n.report=n.report, return.neighbor.info=TRUE)

# predict the latent method
p.r <- predict(sim.r, X.0 = x.ho, coords.0 = coords.ho, n.omp.threads=1)

p_hat = apply(p.r$p.y.0, 1, mean)

write.csv(p_hat, file.y.ho)


# summary of the latent method
summary(sim.r)
