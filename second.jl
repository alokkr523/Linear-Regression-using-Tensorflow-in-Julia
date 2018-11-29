using DataFrames
using GLM

train = readtable("trainSet.csv")
test = readtable("testSet2.csv")
train[:KWH]
#X_train = train[:, filter(x -> x != :KWH, names(train))]
X_test = test[:, filter(x -> x != :KWH, names(test))]
#y_train = train[:, filter(x -> x == :KWH, names(train))]
y_test = test[:, filter(x -> x == :KWH, names(test))]

#y_train = convert(Array, train[:, filter(x -> x == :KWH, names(train))])
#X_train = convert(Array, train[:, filter(x -> x != :KWH, names(train))])
#y_test = convert(Array, test[:, filter(x -> x == :KWH, names(test))])
#X_test = convert(Array, test[:, filter(x -> x != :KWH, names(test))])


OLS = glm(@formula(KWH ~ YEARMADERANGE + WALLTYPE + BEDROOMS + NCOMBATH + NHAFBATH + STOVENFUEL + OVENUSE + AMTMICRO + TOASTER + NUMFRIG + NUMFREEZ + DWASHUSE + WASHLOAD + DRYRUSE + TVCOLOR + TVSIZE1 + TVONWD1 + NUMPC + TIMEON1 + TIMEON2 + WELLPUMP + AQUARIUM + BATTOOLS + HEATROOM + MOISTURE + COOLTYPE + USECENAC + PROTHERMAC + NUMCFAN  + TREESHAD + RECBATH + LIGHTS + SLDDRS + ADQINSUL + USENG + ELWARM + ELCOOL + ELWATER + ELFOOD + EDUCATION + NHSLDMEM + MEANAGE + MONEYPY + POVERTY150 + TOTSQFT + TOTCSQFT + WSF), train, Normal(), IdentityLink())

y_pred = predict(OLS, X_test)

OLS2 = fit(LinearModel,@formula(KWH ~ YEARMADERANGE + WALLTYPE + BEDROOMS + NCOMBATH + NHAFBATH + STOVENFUEL + OVENUSE + AMTMICRO + TOASTER + NUMFRIG + NUMFREEZ + DWASHUSE + WASHLOAD + DRYRUSE + TVCOLOR + TVSIZE1 + TVONWD1 + NUMPC + TIMEON1 + TIMEON2 + WELLPUMP + AQUARIUM + BATTOOLS + HEATROOM + MOISTURE + COOLTYPE + USECENAC + PROTHERMAC + NUMCFAN  + TREESHAD + RECBATH + LIGHTS + SLDDRS + ADQINSUL + USENG + ELWARM + ELCOOL + ELWATER + ELFOOD + EDUCATION + NHSLDMEM + MEANAGE + MONEYPY + POVERTY150 + TOTSQFT + TOTCSQFT + WSF),train)

r2(OLS2)
adjr2(OLS2)

OLS2[:Estimate]

t = [ccdf(FDist(1,df_residual(OLS2.model)),abs2(fval)) for fval in coef(OLS2)./stderr(OLS2)]
t = [ccdf(FDist(1,df_residual(OLS.model)),abs2(fval)) for fval in coef(OLS)./stderr(OLS)]

v = t
maxval = maximum(v)
positions = [i for (i, x) in enumerate(v) if x == maxval]
