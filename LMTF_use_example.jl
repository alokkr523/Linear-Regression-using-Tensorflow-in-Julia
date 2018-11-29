
using TensorFlow, CSV, DataFrames
cd(raw"/home/alok/Desktop/Blackcoffer/Julia+Tensor Flow/Data")
include("LMTF.jl")
Train = CSV.read("trainSet.csv");
Test = CSV.read("testSet.csv");

X_train = DataFrame(YEARMADERANGE = (Train[:YEARMADERANGE]), WALLTYPE = categorical(Train[:WALLTYPE]),
    ROOFTYPE = categorical(Train[:ROOFTYPE]),BEDROOMS = (Train[:BEDROOMS]), NCOMBATH = (Train[:NCOMBATH]),
    NHAFBATH = (Train[:NHAFBATH]), PRKGPLC1 = categorical(Train[:PRKGPLC1]), STOVENFUEL = (Train[:STOVENFUEL]),
    OVENUSE = (Train[:OVENUSE]),  AMTMICRO = (Train[:AMTMICRO]), DEFROST = categorical(Train[:DEFROST]),
    TOASTER = categorical(Train[:TOASTER]), NUMMEAL = (Train[:NUMMEAL]), FUELFOOD = categorical(Train[:FUELFOOD]) ,
    COFFEE = categorical(Train[:COFFEE]) , NUMFRIG = (Train[:NUMFRIG]) ,TYPERFR1 = categorical(Train[:TYPERFR1]) , 
    SIZRFRI1 = (Train[:SIZRFRI1]) , AGERFRI1 = (Train[:AGERFRI1]) , NUMFREEZ = (Train[:NUMFREEZ]) , 
    DWASHUSE = (Train[:DWASHUSE]) , WASHLOAD = (Train[:WASHLOAD]) , DRYRUSE = (Train[:DRYRUSE]) , 
    TVCOLOR = (Train[:TVCOLOR]) , TVSIZE1 = (Train[:TVSIZE1]) , TVTYPE1 = categorical(Train[:TVTYPE1]) , 
    TVONWD1 = (Train[:TVONWD1]) , TVONWE1 = (Train[:TVONWE1]) , NUMPC = (Train[:NUMPC]) ,
    TIMEON1 = (Train[:TIMEON1]) , TIMEON2 = (Train[:TIMEON2]), TIMEON3 = (Train[:TIMEON3]) , 
    INTERNET = categorical(Train[:INTERNET]) , WELLPUMP = categorical(Train[:WELLPUMP]) ,
    AQUARIUM = categorical(Train[:AQUARIUM]) , STEREO = categorical(Train[:STEREO]) , BATTOOLS = (Train[:BATTOOLS]) , 
    HEATROOM = (Train[:HEATROOM]) , MOISTURE = categorical(Train[:MOISTURE]) ,FUELH2O = categorical(Train[:FUELH2O]) ,
    COOLTYPE = categorical(Train[:COOLTYPE]) , ACROOMS = (Train[:ACROOMS]) , USECENAC = (Train[:USECENAC]) , 
    PROTHERMAC = categorical(Train[:PROTHERMAC]) , NUMCFAN = (Train[:NUMCFAN]) , USECFAN = (Train[:USECFAN]) ,
    TREESHAD = (Train[:TREESHAD]) , RECBATH = categorical(Train[:RECBATH]) , LIGHTS = (Train[:LIGHTS]) , 
    SLDDRS = categorical(Train[:SLDDRS]) , WINDOWS = (Train[:WINDOWS]), ADQINSUL = (Train[:ADQINSUL]) , 
    USENG = categorical(Train[:USENG]) , USESOLAR = categorical(Train[:USESOLAR]) , ELWARM = categorical(Train[:ELWARM]) ,
    ELCOOL = categorical(Train[:ELCOOL]),     ELWATER = categorical(Train[:ELWATER]) , ELFOOD = categorical(Train[:ELFOOD]) ,
    EMPLOYHH = categorical(Train[:EMPLOYHH]) , Householder_Race = categorical(Train[:Householder_Race]) , 
    EDUCATION = categorical(Train[:EDUCATION]) ,NHSLDMEM = (Train[:NHSLDMEM]) , MEANAGE = (Train[:MEANAGE])  , 
    RETIREPY = categorical(Train[:RETIREPY]) , MONEYPY = (Train[:MONEYPY]) ,POVERTY150 = categorical(Train[:POVERTY150]) ,
    TOTSQFT = (Train[:TOTSQFT]) , TOTHSQFT =(Train[:TOTHSQFT]) , TOTCSQFT = (Train[:TOTCSQFT]) , WSF = (Train[:WSF]) );

Y_train = Train[:, filter(x -> x == :KWH, names(Train))];
X_test = Test[:, filter(x -> x != :KWH, names(Test))];
Y_test = Test[:, filter(x -> x == :KWH, names(Test))];

LMTF(X_train, Y_train, X_test, Y_test);


