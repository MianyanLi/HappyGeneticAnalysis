# ideal-spoon
Codes for conducting genomic prediction under the non-additive genetic background

# 1 python codes for KPRR and SPVR
import pandas as pd
import numpy as np
import time as time
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error
import os

os.chdir('your path')

X = pd.read_csv('gen.txt', sep='\t', header=None)
y = pd.read_csv('phe.txt', sep='\t', header=None)

y = y[1]

X = np.asarray(X)
y = np.asarray(y)

# grid search
krr_param = {'kernel': ['poly'], 'alpha': [10, 1, 0.1, 0.01], 'gamma': [10, 1, 100]}
kfold = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
krr_gs = GridSearchCV(estimator=KernelRidge(),
                      param_grid=krr_param, cv=5,
                      scoring='neg_mean_squared_error').fit(X, y)

cv_results = pd.DataFrame(krr_gs.cv_results_)

# best parameters
col = cv_results[cv_results.rank_test_score == 1].index.tolist()[0]
best = pd.DataFrame(cv_results.loc[col])
best.columns = ['0']

alpha = best.loc['param_alpha'][0]
gamma = best.loc['param_gamma'][0]

# repeat for the best parameter
res = pd.DataFrame(np.zeros([25, 2]))
res.columns = ['corr', 'mse']
k = 0

start = time.time()
krr = KernelRidge(kernel='poly', alpha=alpha, gamma=gamma)
kfold = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
for train_index, test_index in kfold.split(X):
    X_train, y_train = X[train_index, :], y[train_index]
    X_test, y_test = X[test_index, :], y[test_index]
    krr.fit(X_train, y_train)
    y_hat = krr.predict(X_test)
    corr = np.corrcoef(y_test, y_hat)[0, 1]
    mse = mean_squared_error(y_test, y_hat)
    res.iloc[k, :] = [corr, mse]
    k += 1
res = res.iloc[:k, :]
end = time.time()

print("Running time: %s seconds" % (end - start))
print(np.mean(res))
print(np.std(res))

outputpath = 'path'
res.to_excel(outputpath, index=False, header=True)


# 2
import pandas as pd
import numpy as np
import time as time
from sklearn.kernel_ridge import SVR
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error
import os

os.chdir('your path')

X = pd.read_csv('gen.txt', sep='\t', header=None)
y = pd.read_csv('phe.txt', sep='\t', header=None)

y = y[1]

X = np.asarray(X)
y = np.asarray(y)

# grid search
krr_param = {'kernel': ['poly'], 'C': [10, 1, 100, 1000], 'gamma': [10, 1, 100, 1000]}
kfold = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
krr_gs = GridSearchCV(estimator=SVR(),
                      param_grid=krr_param, cv=5,
                      scoring='neg_mean_squared_error').fit(X, y)

cv_results = pd.DataFrame(krr_gs.cv_results_)

col = cv_results[cv_results.rank_test_score == 1].index.tolist()[0]
best = pd.DataFrame(cv_results.loc[col])
best.columns = ['0']

C = best.loc['param_C'][0]
gamma = best.loc['param_gamma'][0]

res = pd.DataFrame(np.zeros([25, 2]))
res.columns = ['corr', 'mse']
k = 0

start = time.time()
svr = SVR(kernel='poly', C=C, gamma=gamma)
kfold = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)

for train_index, test_index in kfold.split(X):
    X_train, y_train = X[train_index, :], y[train_index]
    X_test, y_test = X[test_index, :], y[test_index]
    svr.fit(X_train, y_train)
    test_pre = svr.predict(X_test)
    corr = np.corrcoef(y_test, test_pre)[0, 1]
    mse = np.sum((y_test - test_pre) ** 2) / len(y_test)
    res.iloc[k, :] = [corr, mse]
    k = k + 1

res = res.iloc[:k, :]
end = time.time()

print("Running time: %s seconds" % (end - start))
print(np.mean(res))
print(np.std(res))

outputpath = 'path'
res.to_excel(outputpath, index=False, header=True)


# 3 R codes for BayesB
rm(list=ls())
library(caret) 
library(BGLR)  
library(rrBLUP)  

setwd('path')
x = read.table("gen.txt", header=F, sep="\t")
y = read.table("phe.txt", header=F, sep="\t")

x = as.matrix(x)
y = as.matrix(y)

G = A.mat(x)

set.seed(12345)
folds = createMultiFolds(y=y, k=5, times=5)

# specify arguments
nIter=20000;
burnIn=10000; 
thin=10;
saveAt='';
S0=NULL;
weights=NULL;
R2=0.5
# specify regression predictor ##
ETA=list(list(X=x,model='BayesB',probIn=0.05),
         list(K=G,model='RKHS'))

res = matrix(nrow=25, ncol=2)

t1=proc.time()
## cross validation ##
for(i in 1:25){
  v = -folds[[i]]
  pheno = y
  pheno[v,] = c('NA')
  pheno = as.matrix(as.numeric(unlist(pheno)))
  #
  model = BGLR(y=pheno,ETA=ETA,nIter=nIter,burnIn=burnIn,thin=thin,saveAt=saveAt,
               df0=5,S0=S0,weights=weights,R2=R2,verbose=FALSE)
  
  yHat = as.data.frame(model$yHat)
  y = as.data.frame(y)
  
  res[i,1] = cor(yHat[v,1], y[v,1], use = 'complete')
  res[i,2] = mean((yHat[v,1] - y[v,1])^2)
}
t2=proc.time()
t=t2-t1

t
summary(res)
write.csv(res,file='path',row.names=F)


# 4 R codes for GBLUP
rm(list=ls())
library(caret) 
library(BGLR) 
library(rrBLUP) 

setwd('path')
x = read.table("gen.txt", header=F, sep="\t")
y = read.table("gen.txt", header=F, sep="\t")

y = y['V1']

colnames(y) = paste0("y",1:ncol(y))

x = as.matrix(x)
y = as.data.frame(y)

y$id = as.factor(rownames(y))
rownames(x) = rownames(y)

G = A.mat(x)

colnames(G) = rownames(G) = rownames(y)

res = matrix(nrow=25, ncol=2)
set.seed(1)
folds = createMultiFolds(y=y[,"y1"], k=5, times=5)

t1=proc.time()
system.time(
  for(i in 1:25){
    y.trn = y
    a = as.character(folds[[i]])
    b = as.character(y[,'id'])
    c = setdiff(b,a)
    y.trn[c,"y1"] = NA
    
    system.time(
      ans <- mmer(y1~1, 
                  random=~vs(id,Gu=G),
                  rcov=~units,
                  data=y.trn, verbose = FALSE))
    y_hat = as.data.frame(ans$U$`u:id`$y1)
    res[i,1] = cor(y_hat[c,], y[c,"y1"], use = 'complete')
    res[i,2] = mean((y_hat[c,])-(y[c,"y1"])^2)
  }
)
t2=proc.time()
t=t2-t1
t
summary(res)
write.csv(res, file='path', row.names=FALSE)


