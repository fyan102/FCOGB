library(pROC)
rsq <- function(x, y) summary(lm(x~y))$r.squared
# rsq = function(y_actual,y_predict){
#   cor(y_actual,y_predict)^2
# }
evaluate.sirus <-
  function(dataset_name,
           file_name,
           attributes,
           target,
           feature_map = list, 
           cr = "r", mdepth=2) {
    start_time=Sys.time()
    record_file<-paste('./R_test/output/',dataset_name,'.txt')
    print(dataset_name)
    risks <- c()
    test_risks <- c()
    train_scores<-c(0)
    test_scores<-c(0)
    components <- c()
    data <- read.csv(file = file_name)
    smp_size <- floor(0.8 * nrow(data))
    train_ind<-sample(seq_len(nrow(data)), size = smp_size)
    train_data<-data[train_ind, ]
    test_data<-data[-train_ind, ]
    set.seed(seed = 0)
    predictions=numeric(nrow(train_data))
    predictions2=numeric(nrow(test_data))
    for (k in names(feature_map)) {
      train_data[train_data == k] <- feature_map[k]
      test_data[test_data==k]<-feature_map[k]
    }
    
    train_data <- type.convert(train_data, as.is = TRUE)
    train_data <- na.omit(train_data)
    y <- train_data[, target]
    test_data <- type.convert(test_data, as.is = TRUE)
    test_data <- na.omit(test_data)
    yy <- test_data[, target]
    # print(y)
    # a = mean(y)
    # b = sd(y)
    # if (cr=='r'){
    #   y<-(y-a)/b
    # }
    # a2 = mean(yy)
    # b2 = sd(yy)
    # if (cr=='r'){
    #   yy<-(yy-a2)/b2
    # }
    for (col in colnames(train_data)) {
      if (!is.element(col, attributes)) {
        train_data[col] <- NULL
      }
    }
    for (col in colnames(test_data)) {
      if (!is.element(col, attributes)) {
        test_data[col] <- NULL
      }
    }
    # aa = colMeans(train_data)
    # bb = apply(train_data, 2, sd)
    # train_data<-(train_data-aa)/bb
    # aa2 = colMeans(test_data)
    # bb2 = apply(test_data, 2, sd)
    # test_data<-(test_data-aa2)/bb2
    # print(str(train_data))
    if (cr == "c") {
      risk <- (sum(log2(1 + exp(
        -y * (predictions)
      ))) / nrow(train_data))
      risk2 <- (sum(log2(1 + exp(
        -yy * (predictions2)
      ))) / nrow(test_data))
    } else {
      risk <- (sum((y - predictions) ** 2) / nrow(train_data))
      risk2 <- (sum((yy - predictions2) ** 2) / nrow(test_data))
      # risk <- (sum((y*b+a - predictions*b) ** 2) / nrow(train_data))
    }
    risks <- append(risks, risk)
    test_risks <- append(test_risks, risk2)
    require(sirus)
    
    # print(mdepth)
    for (i in 1:51) {
      sirus.m <-
        sirus.fit(
          train_data,
          y,
          num.rule = i,
          discrete.limit = 3,
          max.depth = mdepth,
          num.trees = 1000
        )
      predictions <- sirus.predict(sirus.m, train_data)
      predictions2<-sirus.predict(sirus.m, test_data)
      
      # if (i==1){
      #   predictions<-rep(c(predictions[1]), each=nrow(train_data))
      #   predictions2<-rep(c(predictions[1]), each=nrow(test_data))
      #   if (cr=="c"){
      #     # roc_object <- roc( y, predictions)
      #     # train_score <- auc(roc_object)
      #     # roc_object2 <- roc( yy, predictions2)
      #     # test_score<-auc(roc_object2)
      #     # print(predictions)
      #     # print(y)
      #     roc_obj <- roc(y, predictions)
      #     train_score<-auc(roc_obj)
      #     roc_obj2 <- roc(yy, predictions2)
      #     test_score<-auc(roc_obj2)
      #   }else{
      #     # print(predictions)
      #     # print(y)
      #     train_score<-rsq(y, predictions)
      #     test_score<-rsq(yy, predictions2)
      #   }
      #   train_scores<-append(train_scores, train_score)
      #   test_scores<-append(test_scores, test_score)
      #   # print(c("train score:",str(train_score)))
      #   # print(c("test_score:",str(test_score)))
      #   write(c("train_score", train_score), file = record_file,append=TRUE)
      #   write(c("test_score", test_score), file = record_file,append=TRUE)
      # }
      if (cr=="c"){
        # roc_object <- roc( y, predictions)
        # train_score <- auc(roc_object)
        # roc_object2 <- roc( yy, predictions2)
        # test_score<-auc(roc_object2)
        # print(predictions)
        # print(y)
        roc_obj <- roc(y, predictions)
        train_score<-auc(roc_obj)
        roc_obj2 <- roc(yy, predictions2)
        test_score<-auc(roc_obj2)
      }else{
        # print(predictions)
        # print(y)
        train_score<-rsq(y, predictions)
        test_score<-rsq(yy, predictions2)
      }
      train_scores<-append(train_scores, train_score)
      test_scores<-append(test_scores, test_score)
      # print(c("train score:",str(train_score)))
      # print(c("test_score:",str(test_score)))
      write(c("train_score", train_score), file = record_file,append=TRUE)
      write(c("test_score", test_score), file = record_file,append=TRUE)
      print(sirus.print(sirus.m))
      write(sirus.print(sirus.m), file = record_file,append=TRUE)
      if (cr == "c") {
        risk <- (sum(log2(1 + exp(
          -y * (predictions)
        ))) / nrow(train_data))
        risk2 <- (sum(log2(1 + exp(
          -yy * (predictions2)
        ))) / nrow(test_data))
      } else {
        # risk <- (sum((y*b - predictions*b) ** 2) / nrow(train_data))
        risk <- (sum((y - predictions) ** 2) / nrow(train_data))
        risk2 <- (sum((yy - predictions2) ** 2) / nrow(test_data))
      }
      risks <- append(risks, risk)
      test_risks<-append(test_risks, risk2)
      cnt <- 0
      for (rule in sirus.m$rules){
        cnt <- length(rule)+2
      }
      
      components <- append(components, cnt)
      print(risk)
      print(risk2)
      write(c('risk: ', risk), file = record_file,append=TRUE)
    }
    risks<-head(risks, -1)
    print(risks)
    print(test_risks)
    print(components)
    write(c('risks: ', risks, '\n','components', components), file = record_file,append=TRUE)
    sum_risk=0
    sum_comp=0
    sum_risk2=0
    sum_train_score=0
    sum_test_score=0
    print(train_scores)
    for (i in 1:51){
      print(risks[i])
      sum_risk<-sum_risk+components[i]*risks[i]
      sum_risk2<-sum_risk2+components[i]*test_risks[i]
      sum_train_score<-sum_train_score+components[i]*train_scores[i]
      sum_test_score<-sum_test_score+components[i]*test_scores[i]
      sum_comp<-sum_comp+components[i]
      if (sum_comp>50){
        break
      }
    }
    # print(sum(risks*components)/sum(components))
    print(sum_risk/sum_comp)
    print(sum_risk2/sum_comp)
    print(sum_train_score/sum_comp)
    print(sum_test_score/sum_comp)
    write(c('average train risk',sum_risk/sum_comp), file = record_file,append=TRUE)
    write(c('average test risk',sum_risk2/sum_comp), file = record_file,append=TRUE)
    # write(c('average train score',sum_train_score/sum_comp), file = record_file,append=TRUE)
    # write(c('average test score',sum_test_score/sum_comp), file = record_file,append=TRUE)
    # close(record_file)
    end_time=Sys.time()
    running_time=end_time-start_time
    write(c('running time', running_time), file=record_file, append=TRUE)
  }

evaluate.sirus(
  "gdp",
  "./R_test/datasets/gdp_vs_satisfaction/GDP_vs_Satisfaction.csv",
  list("GDP"),
  "Satisfaction",
  list(), mdepth = 2,
)
evaluate.sirus(
  "titanic",
  "./R_test/datasets/titanic/train.csv",
  c('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'),
  "Survived",
  c(
    'male' = 1,
    'female' = 0,
    'S' = 1,
    'C' = 2,
    'Q' = 3
  ),
  "c"
)
evaluate.sirus(
  'wage',
  './R_test/datasets/wages_demographics/wages.csv',
  c('height', 'sex', 'race', 'ed', 'age'),
  'earn',
  c(
    'male' = 1,
    'female' = 0,
    'white' = 1,
    'black' = 2,
    'hispanic' = 3,
    'other' = 4
  )
)
evaluate.sirus(
  'insurance',
  './R_test/datasets/insurance/insurance.csv',
  c('age', 'sex', 'bmi', 'children', 'smoker', 'region'),
  'charges',
  feature_map = c(
    'male' = 1,
    'female' = 0,
    'yes' = 1,
    'no' = 0,
    'southwest' = 1,
    'southeast' = 2,
    'northwest' = 3,
    'northeast' = 4
  )
)
evaluate.sirus(
  'world_happiness_indicator',
  './R_test/datasets/world_happiness_indicator/2019.csv',
  c(
    'GDP per capita',
    'Social support',
    'Healthy life expectancy',
    'Freedom to make life choices',
    'Generosity',
    'Perceptions of corruption'
  ),
  'Score',
  c()
)
evaluate.sirus(
  'Demographics',
  './R_test/datasets/Demographics/Demographics1.csv',
  c(
    'Sex',
    'Marital',
    'Age',
    'Edu',
    'Occupation',
    'LivingYears',
    'Persons',
    'PersonsUnder18',
    'HouseholderStatus',
    'TypeOfHome',
    'Ethnic',
    'Language'
  ),
  'AnnualIncome',
  c(' Male'= 1, ' Female'=0,
    ' Married' = 1,
    ' Single, never married' = 2,
    ' Divorced or separated' = 3,
    ' Living together, not married' = 4,
    ' Widowed' = 5,
    ' Homemaker' = 1,
    ' Professional/Managerial' = 2,
    ' Student, HS or College' = 3,
    ' Retired' = 4,
    ' Unemployed' = 5,
    ' Factory Worker/Laborer/Driver' = 6,
    ' Sales Worker' = 7,
    ' Clerical/Service Worker' = 8,
    ' Military' = 9,
    ' Own'=1,
    ' Rent'=2,
    ' Live with Parents/Family'=3,
    ' House'=1,
    ' Apartment'=2,
    ' Condominium'=3,
    ' Mobile Home'=4,
    ' White'=1,
    ' Hispanic'=2,
    ' Asian'=3,
    ' Black'=4,
    ' East Indian'=5,
    ' Pacific Islander'=6,
    ' American Indian'=7,
    ' Other'=8,
    ' English'=1,
    ' Spanish'=2
  )
)
evaluate.sirus('IBM_HR',
                 './R_test/datasets/IBM_HR/WA_Fn-UseC_-HR-Employee-Attrition.csv',
                 c("Age", 'BusinessTravel', 'DailyRate', 'Department',
                  'DistanceFromHome',
                  'Education',
                  'EducationField', 'EnvironmentSatisfaction', 'Gender',
                  'HourlyRate',
                  'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
                  'MaritalStatus',
                  'MonthlyIncome',
                  'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
                  'PercentSalaryHike',
                  'PerformanceRating', 'RelationshipSatisfaction',
                  'StockOptionLevel',
                  'TotalWorkingYears', 'TrainingTimesLastYear',
                  'WorkLifeBalance',
                  'YearsAtCompany',
                  'YearsInCurrentRole', 'YearsSinceLastPromotion',
                  'YearsWithCurrManager'),
                 'Attrition',
                 feature_map=c('Travel_Rarely'= 1,
                     'Travel_Frequently'= 2,
                     'Non-Travel'= 3, 'Yes'= 1, 'No'= 0,
                     'Sales'= 1, 'Research & Development'= 2,
                     'Human Resources'= 3,'Life Sciences'= 1, 'Medical'= 2,
                     'Marketing'= 3,
                     'Technical Degree'= 4,
                     'Human Resources'= 5,
                     'Other'= 6,'Male'= 1, 'Female'= 0,'Sales Executive'= 1, 'Research Scientist'= 2,
                     'Laboratory Technician'= 3,
                     'Manufacturing Director'= 4,
                     'Healthcare Representative'= 5,
                     'Manager'= 6, 'Human Resources'= 7,
                     'Research Director'= 8,
                     'Sales Representative'= 9,'Single'= 1, 'Married'= 2,
                     'Divorced'= 3 ),cr="c")
evaluate.sirus('make_friedman1', './R_test/datasets/make_friedman1/make_friedman1.csv',
               c('x1', 'x2', 'x3',	'x4', 'x5', 'x6', 'x7','x8','x9','x10'),'y')
evaluate.sirus('make_friedman2', './R_test/datasets/make_friedman2/make_friedman2.csv',
               c('x1', 'x2', 'x3',	'x4'),'y')
evaluate.sirus('make_Friedman3', './R_test/datasets/make_friedman3/make_friedman3.csv',
               c('x1', 'x2', 'x3',	'x4'),'y')
evaluate.sirus('breast_cancer', './R_test/datasets/breast_cancer/breast_cancer.csv',
               c('mean.radius','mean.texture','mean.perimeter','mean.area','mean.smoothness',
                 'mean.compactness','mean.concavity','mean.concave.points','mean.symmetry',
                 'mean.fractal.dimension','radius.error','texture.error','perimeter.error','area.error',
                 'smoothness.error','compactness.error','concavity.error','concave.points.error','symmetry.error',
                 'fractal.dimension.error','worst.radius','worst.texture','worst.perimeter','worst.area',
                 'worst.smoothness','worst.compactness','worst.concavity','worst.concave.points','worst.symmetry',
                 'worst.fractal.dimension'),'y', cr="c")
evaluate.sirus('iris', './R_test/datasets/iris/iris.csv',
               c('sepal.length..cm.','sepal.widt..cm.','petal.lengt..cm.','petal.width..cm.'),'y',cr='c')
evaluate.sirus('load_diabetes', './R_test/datasets/load_diabetes/load_diabetes.csv',
               c('age','sex','bmi','bp','s1','s2','s3','s4','s5','s6'), 'y')
evaluate.sirus("load_wine", './R_test/datasets/load_wine/load_wine.csv',
               c('alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols',
                 'flavanoids','nonflavanoid_phenols','proanthocyanins','color_intensity','hue',
                 'od280/od315_of_diluted_wines','proline'),'y',cr='c')
evaluate.sirus('used_cars',
                 './R_test/datasets/used_cars/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv',
                 c('count', 'km', 'year', 'powerPS'), 'avgPrice')
evaluate.sirus('tic-tac-toe', './R_test/datasets/tic_tac_toe/tic_tac_toe.csv',
               c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"),
                'V10',
               feature_map=c('x'= 1, 'o'= 2, 'b'= 3,'positive'= 1, 'negative'= -1), cr='c')
evaluate.sirus('boston', './R_test/datasets/boston/boston_house_prices.csv',
               c('CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
                'B','LSTAT'),
               'MEDV',c())
evaluate.sirus('banknote', './R_test/datasets/banknotes/banknote.csv',
               c('variance', 'skewness', 'curtosis', 'entropy'), 'class',
               feature_map=c(), cr="c")
evaluate.sirus('liver', './R_test/datasets/liver/liver.csv',
               c('mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks'),
               'selector',
               feature_map=c('1'= 1, '2'= -1), cr='c')
evaluate.sirus('magic', './R_test/datasets/magic/magic04.csv',
               c('fLen1t-1', 'fWidt-1', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Lon1',
                'fM3Trans', 'fAlp-1a', 'fDist'),
               'class',c(), cr="c")
evaluate.sirus('adult', './R_test/datasets/adult/adult.csv',
               c('age', 'workclass', 'fnlwgt', 'education-num', 'marital-status',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week'),
               'output',
               feature_map=c('Private'= 0, 'Self-emp-not-inc'= 1, 'Self-emp-inc'= 2,
                   'Federal-gov'= 3, 'Local-gov'= 4, 'State-gov'= 5,
                   'Without-pay'= 6, 'Never-worked'= 7,'Married-civ-spouse'= 0, 'Divorced'= 1,
                   'Never-married'= 2, 'Separated'= 3, 'Widowed'= 4,
                   'Married-spouse-absent'= 5, 'Married-AF-spouse'= 6,'Wife'= 0, 'Own-child'= 1, 'Husband'= 2,
                   'Not-in-family'= 3, 'Other-relative'= 4, 'Unmarried'= 5,'White'= 0, 'Asian-Pac-Islander'= 1,
                   'Amer-Indian-Eskimo'= 2,
                   'Other'= 3, 'Black'= 4)
               ,cr="c")
evaluate.sirus(
  'GenderRecognition',
  './R_test/datasets/GenderRecognition/voice.csv',
  c("meanfreq", "sd", "median", "Q25", "Q75", "IQR",
   "skew", "kurt",
   "sp.ent", "sfm", "mode", "centroid", "meanfun",
   "minfun", "maxfun",
   "meandom", "mindom", "maxdom", "dfrange", "modindx"),
  "label",
  feature_map=c('male'= 1, 'female'= -1),cr="c")
evaluate.sirus('mobile_prices',
                 './R_test/datasets/mobile_prices/train.csv',
                 c('battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',
                  'four_g',
                  'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc',
                  'px_height',
                  'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
                  'touch_screen',
                  'wifi'), 'price_range', c())

evaluate.sirus('telco_churn',
                 './R_test/datasets/telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',
                 c('gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                  'PhoneService', 'MultipleLines', 'InternetService',
                  'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV',
                  'StreamingMovies', 'Contract', 'PaperlessBilling',
                  'PaymentMethod',
                  'MonthlyCharges', 'TotalCharges' ),
                 'Churn',
                 feature_map=c('Male'= 1, 'Female'= 0, 'Yes'= 1, 'No'= 0,
                     'No phone service'= 3,'DSL'= 1, 'Fiber optic'= 2,
                     'No internet service'= 3,'Month-to-month'= 1, 'One year'= 2,
                     'Two year'= 3,'Electronic check'= 1,
                     'Mailed check'= 2,
                     'Bank transfer (automatic)'= 3,
                     'Credit card (automatic)'= 4
                 ), cr="c")

evaluate.sirus('red_wine_quality', './R_test/datasets/red_wine_quality/winequality-red.csv',
               c('fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides',
                 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates',
                 'alcohol'),
               'quality', c())

evaluate.sirus('who_life_expectancy', './R_test/datasets/who_life_expectancy/Life Expectancy Data.csv',
                 c('Year', 'Status', 'Adult.Mortality', 'infant.deaths', 'Alcohol',
                  'percentage.expenditure',
                  'Hepatitis.B', 'Measles', 'BMI', 'under.five.deaths', 'Polio',
                  'Total.expenditure',
                  'Diphtheria', ' HIV.AIDS', 'GDP', 'Population', 'thinness..1.19.years',
                  ' thinness.5.9.years', 'Income.composition.of.resources', 'Schooling'),
                 'Life.expectancy',
                 feature_map=c('Developing'= 0, 'Developed'=1),
                 )

evaluate.sirus('suicide_rates_cleaned', './R_test/datasets/suicide_rates_cleaned/master.csv',
                 c('year', 'sex', 'age', 'population', 'gdp_for_year....', 'gdp_per_capita....',
                  'generation'),
                 'suicides.100k.pop', c())

evaluate.sirus('videogamesales', './R_test/datasets/videogamesales/vgsales.csv',
                 c('Platform', 'Genre'), 'Global_Sales', c())
evaluate.sirus('digits5', './R_test/datasets/digits/digits.csv',
               c('pixel_0_0',  'pixel_1_0',  'pixel_2_0',  'pixel_3_0',  'pixel_4_0',  'pixel_5_0',  'pixel_6_0',  'pixel_7_0',  'pixel_0_1',  'pixel_1_1',  'pixel_2_1',  'pixel_3_1',  'pixel_4_1',  'pixel_5_1',  'pixel_6_1',  'pixel_7_1',  'pixel_0_2',  'pixel_1_2',  'pixel_2_2',  'pixel_3_2',  'pixel_4_2',  'pixel_5_2',  'pixel_6_2',  'pixel_7_2',  'pixel_0_3',  'pixel_1_3',  'pixel_2_3',  'pixel_3_3',  'pixel_4_3',  'pixel_5_3',  'pixel_6_3',  'pixel_7_3',  'pixel_0_4',  'pixel_1_4',  'pixel_2_4',  'pixel_3_4',  'pixel_4_4',  'pixel_5_4',  'pixel_6_4',  'pixel_7_4',  'pixel_0_5',  'pixel_1_5',  'pixel_2_5',  'pixel_3_5',  'pixel_4_5',  'pixel_5_5',  'pixel_6_5',  'pixel_7_5',  'pixel_0_6',  'pixel_1_6',  'pixel_2_6',  'pixel_3_6',  'pixel_4_6',  'pixel_5_6',  'pixel_6_6',  'pixel_7_6',  'pixel_0_7',  'pixel_1_7',  'pixel_2_7',  'pixel_3_7',  'pixel_4_7',  'pixel_5_7',  'pixel_6_7',  'pixel_7_7'), 
               'target',
               c( '0'=-1, '1'=-1, '2'=-1, '3'=-1, '4'=-1, '6'=-1,
                 '7'=-1, '8'=-1, '9'=-1, '5'=1))

