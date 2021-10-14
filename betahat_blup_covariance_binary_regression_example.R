## --------------------------------------------------------------------------------------------------------------------------------------

setwd("/Users/raswa/Dropbox/nsf_art/article_logistic_backfitting_material/")
source("betahat_blup_covariance_binary_regression.R")

df = readRDS("sample_data.rds")

Y = df$y
X = df[,4:ncol(df)]

F1 = factor(df$factor1)
F2 = factor(df$factor2)

NF1 = length(unique(F1))
NF2 = length(unique(F2))

cat("N is : ",length(Y), "R is ",NF1, "C is ",NF2)

## --------------------------------------------------------------------------------------------------------------------------------------
#get parameter estimate 
start_time_fit_space = Sys.time()
schall_logistic_v2 = 
  backfitting_outer_loop(as.matrix(X),Y,F1,F2,betahat = rep(0,(ncol(X))),sigma_a_sqhat=1,sigma_b_sqhat=1,epsilon=1e-4,max_iter=100,over_dispersed=TRUE,trace.it=TRUE,epsilon_backfitting=1e-8,inner_trace.it=TRUE)
end_time_fit_space = Sys.time()

cat("time taken : ",difftime(end_time_fit_space,start_time_fit_space,units="mins"))
glm_mod = glm(Y~-1+as.matrix(X),family="binomial")

## --------------------------------------------------------------------------------------------------------------------------------------

#obtain cov(\hat{\beta}_{glmm}) under glmm

var_schall = var_betahat_glmm(as.matrix(X),F1,F2,trace.it=TRUE,
                                           schall_logistic_v2)

plogistic = glm_mod$fitted.values
wlogistic = as.vector(plogistic*(1-plogistic))

#obtain obtain cov(\hat{\beta}_{logistic}) under logistic and cov(\hat{\beta}_{logistic}) under glmm
cov_betahat_logistic = solve(t(as.matrix(X))%*%(wlogistic*as.matrix(X)))
covzx = var_betahat_logistic_under_glmm(as.matrix(X),F1,F2,trace.it=TRUE,schall_logistic_v2,var_schall,wlogistic)$cov_zx
cov_betahat_logistic_under_glmm = solve(t(as.matrix(X))%*%(wlogistic*as.matrix(X)))%*%(covzx)%*%solve(t(as.matrix(X))%*%(wlogistic*as.matrix(X)))


#naivete
hist(diag(cov_betahat_logistic_under_glmm)/diag(vcov(glm_mod)),col="steelblue",
     main="Naivety of Logistic by Coefficient",xlab="",breaks=10)#breaks=eigv[c(1,10,20,34)]

#inefficiency
hist(diag(cov_betahat_logistic_under_glmm)/diag(var_schall$cov),col="steelblue",
     main="Inefficiency of Logistic by Coefficient",xlab="",breaks=10)#breaks=eigv[c(1,10,20,34)]


