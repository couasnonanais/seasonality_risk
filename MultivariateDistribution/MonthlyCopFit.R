rm(list=ls(all=TRUE))
library(VineCopula)
library(copula)
library(mvtnorm)
library(MASS)
library(LaplacesDemon)
library(ggplot2)
library(reshape2)

myarg <- commandArgs()
month = as.integer(myarg[6])
n = as.integer(myarg[7])
print(month)
print(n)

varname1 = myarg[8]
varname2 = myarg[9]
print(varname1)
print(varname2)

#print(myarg)

# Load data from file location
trial <- read.csv(file.path('E:/surfdrive/Documents/Master2019/Thomas/data/NewBivariate', paste('New',varname1,'_New',varname2, '_data_month_',month,'.csv',  sep='')),header=T)

print(varname1)
print(varname2)
print(month)

#Change depending on event selection
var1 <- trial[, varname1] #Rainfall
var2 <- trial[, varname2] #SWL
ori <- as.matrix(cbind(var1, var2))

ind1 <- sample(var1)
ind2 <- sample(var2)
ind <- as.matrix(cbind(ind1,ind2))

#Make pseudo-observations
u <- pobs(as.matrix(cbind(var1,var2)))
pairs(u)  # plot observations

u_ind <- pobs(as.matrix(cbind(ind1,ind2)))
pairs(u_ind)

#Select copula
cl <- BiCopSelect(u1=u[,1],u2=u[,2],familyset=c(0,1,2,3,4,5,6,7,8,9,10), rotations = T) #familyset=c(0,1,3,4,5)
#summary(cl)

#We save the summary of the fitted copula
summary_cop <- as.data.frame(c(cl$familyname, cl$family, cl$taildep$upper,cl$taildep$lower, cl$p.value.indeptest, cl$tau))
rho_rank = BiCopTau2Par(family=1, cl$tau, check.taus = TRUE) # find rank correlation for given value of tau using copula

#We simulate data
data_sim <- BiCopSim(n, obj = cl, check.pars = TRUE)
pairs(data_sim)

#We export the data
write.csv(data_sim,file.path('E:/surfdrive/Documents/Master2019/Thomas/data/NewBivariate','Simulated', paste('New',varname1,'_New',varname2, '_data_month_',month,'.csv',  sep='')), na="nan",row.names=TRUE)
write.csv(summary_cop,file.path('E:/surfdrive/Documents/Master2019/Thomas/data/NewBivariate','Simulated', paste('New',varname1,'_New',varname2, '_copulatype_month_',month,'.csv',  sep='')))