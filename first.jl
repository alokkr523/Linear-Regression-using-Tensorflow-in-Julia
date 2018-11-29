using DataFrames

train = readtable("train.csv")

size(train)
names(train)
head(train, 10)

describe(train[:LoanAmount])

showcols(train)

#replace missing loan amount with mean of loan amount
train[isna.(train[:LoanAmount]),:LoanAmount] = floor(mean(dropna(train[:LoanAmount])))
#replace 0.0 of loan amount with the mean of loan amount
train[train[:LoanAmount] .== 0, :LoanAmount] = floor(mean(dropna(train[:LoanAmount])))
#replace missing gender with mode of gender values
train[isna.(train[:Gender]), :Gender] = mode(dropna(train[:Gender]))
#replace missing married with mode value
train[isna.(train[:Married]), :Married] = mode(dropna(train[:Married]))
#replace missing number of dependents with the mode value
train[isna.(train[:Dependents]),:Dependents]=mode(dropna(train[:Dependents]))
#replace missing values of the self_employed column with mode
train[isna.(train[:Self_Employed]),:Self_Employed]=mode(dropna(train[:Self_Employed]))
#replace missing values of loan amount term with mode value
train[isna.(train[:Loan_Amount_Term]),:Loan_Amount_Term]=mode(dropna(train[:Loan_Amount_Term]))
#replace credit history missing values with mode
train[isna.(train[:Credit_History]), :Credit_History] = mode(dropna(train[:Credit_History]))


using ScikitLearn: fit!, predict, @sk_import, fit_transform!
@sk_import preprocessing: LabelEncoder
labelencoder = LabelEncoder()
categories = [2 3 4 5 6 12 13]

for col in categories
    train[col] = fit_transform!(labelencoder, train[col])
end
