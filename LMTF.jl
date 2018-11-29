
#LMTF definition 
function LMTF(X_train, Y_train, X_test, Y_test; varcount = 0, alpha = 0.0000000001)
    
    # converting Y_train database into array 
    Y_train1 = convert(Array{Float32}, Y_train[:,:]);
    
    # calculating number of rows and column in X_train and X_test
    m = nrow(X_train)
    n = ncol(X_train)
    m1 = nrow(X_test)
    
    # Declaring X_train1 and X_test1 to store and work while dealing with categorical variable
    global X_train1 = zeros(m,0)
    global X_test1 = zeros(m1,0)
    
    # Storing the names of variables in X_train in nm  
    nm = names(X_train);
    
    # Storing the names of variables for categorical variable
    global colname = Array{String,1}() ;
    
    
    for i=1:n
        
        # Checking whether variable is categorical or not
        if (isa(X_train[i], CategoricalArrays.CategoricalArray)) == true
            
            # Storing categorical varible for train and test set in varcol and varcol2
            varcol = X_train[i]
            varcol2 = X_test[i]
            
            # uniq will store unique variables of a single category
            uniq = unique(varcol);
            
            # nmp store the name of current categorical variable
            nmp = nm[i]
            
            # Calculate the number of unique variable in uniq
            N_ind = size(uniq,1);
            

	    # Assigning the name for each unique variable in a category
            for j=1:N_ind
                br = uniq[j]
                str = string(nmp,"_", br)
                push!(colname, str)
            end
            
            #  calculating number of rows in varcol and varcol1
            N_obs = size(varcol,1);
            N_obs2 =size(varcol2,1);
            
            
            # Creating dummy variable for categorical variable
            global D = zeros(N_obs,N_ind);
            global D2 = zeros(N_obs2, N_ind);
            
            # Assigning 0 and 1 value to dummy variable 
            for j=1:N_ind
                D[:,j] = Int.(varcol .== uniq[j,1])
                D2[:,j] = Int.(varcol2 .== uniq[j,1])
            end
            
            # Updating X_train1 and X_test1 by concatenating Dummy variable matrix
            X_train1 = hcat(X_train1,D)
            X_test1 = hcat(X_test1, D2)
        else
            
            # Updating colname, X_train1, X_test1 for non-categorical variable
            push!(colname, nm[i])
            varcol = X_train[i]
            varcol2 = X_test[i]
            X_train1 = hcat(X_train1,varcol)
            X_test1 =hcat(X_test1, varcol2)
        end
    end
    
    # Calculating number of rows and column in X_train1
    r = size(X_train1,1)
    c = size(X_train1,2)
    
    # Converting X_train1 and X_test1 into Array
    X_train1 = convert(Array{Float32}, X_train1[:,:]);
    X_test1 = convert(Array{Float32}, X_test1[:,:]);
    
    # ind_min to calculate the index of minimum t-value intialised with 0
    global ind_min = 0;
    
    # To show all the coeficient when no varcount is passed
    if (varcount == 0)
        varcount = c;
    end
    
    # Calculation of Weights and bias terms
    while(c >= varcount)
        
        #starting Tensorflow session
        tfsession = Session(Graph())
        
        # Placeholder to hold the value of X and Y 
        X = placeholder(Float32, shape=[r,c])
        Y = placeholder(Float32, shape=[r,1])
        
        # Variables to store the Weights and bias
        W = Variable(rand(c, 1) * (-0.2-0.2) - 0.2) 
        b = Variable(rand(1, 1) * (-0.2-0.2) - 0.2)
        
        #linear model
        linear_model = X*W + b
        
        # Cost function or loss from original data and model
        loss = reduce_sum(((linear_model - Y).^2)./(2*r))
        
        # Feeding learning rate alpha and telling algorithm to minimize the loss
        optimize = train.GradientDescentOptimizer(alpha)
        alg = train.minimize(optimize, loss)
        
        # Initialising Variables W and b
        run(tfsession, global_variables_initializer())

        # Running the iteration
        for i in 1:1000
            run(tfsession, alg, Dict(X=>X_train1, Y=>Y_train1))
        end
    
        # Declaring variable to store current Weight, current bias, current loss, current t_stat and current t-stat with bias term
        global curr_W, curr_b, curr_loss, t_statb, t_stat
        
        # Storing the current value of Weights, bias and loss term
        curr_W, curr_b, curr_loss = run(tfsession, [W, b, loss], Dict(X=>X_train1, Y=>Y_train1))


        # Predicted value of Y
        Y_pred = X_train1 * curr_W .+ curr_b
        
        # Sum of square error on train set
        SSE = sum((Y_train1 - Y_pred).^2) 

        # Adding first column with value 1 in X_train1
        X_1 = ([1; (X_train1')])'
        
        # Degree of freedom
        dof = r-c-1 
        
        # Error mean square
        MS = SSE/dof
        
        # Standard error in each variables
        Std_err =(diag(inv(X_1'*X_1))) .* MS
        
        # Weights Including bias term
        Beta = [curr_b; curr_W]
        
        # t_stat including bias term
        t_statb = Beta ./ Std_err
        
        # t_stat without bias term
        t_stat = t_statb[1:end .!=1, :]
        
        # Index of minimum absolute value in t_stat
        ind_min = indmin(abs.(t_stat))
        
        # Decrement the value of c
        c = c-1
        
        # Deleting the column with index ind_min from colname, X_train1, X_test1
        if (c >= varcount)
            colname = deleteat!(colname, ind_min)
            X_train1 = X_train1[:, 1:end .!=ind_min]
            X_test1 = X_test1[:, 1:end .!=ind_min]
        end

    end

    # Print the bias term
    print("\n\nbias term : $(curr_b)\n\n")
    
    # Making a dataframe of column name, weights, t_statistics and displaying it
    daf = DataFrame(Variables = colname, Weights = curr_W[:,1], t_statistics = t_stat[:,1]);
    showall(daf)
    
    #Predicting value of Y on Test set
    Pred_Y = X_test1*curr_W .+ curr_b ;
    
    # Converting it into a data frame
    df = DataFrame(Array(Pred_Y)) ;
    
    # Converting Y_test, Pred_Y to array
    Y_1 = convert(Array, Y_test);
    Y_2 = convert(Array, Pred_Y);
    
    # Calculating Sum of Square error on test set
    SSEtest = sum((Y_1-Y_2).^2)
    
    # Print sum of square error on test set
    print("\n\nSum of Square Error in Test Set : $(SSEtest) \n\n")
    
    # Writing csv file storing the predicted result
    CSV.write("Resultv4.csv", df);
    
end

