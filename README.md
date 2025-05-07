# Outline of Readme.md markdown file: (This constitutes the majority of your documentation)

Aircraft wildlife strikes are a major safety concern in aviation. This project predicts the most likely animal species involved in a strike based on factors like flight phase, location, and altitude.

# Describe the dataset(s) used.

    https://wildlife.faa.gov/assets/fieldlist.pdf

Based on the provided dataset, here's the expected data type for each of the columns:

1. **Record ID**: `Integer` Individual record number
2. **Incident Year**: `Integer` Year strike occurred
3. **Incident Month**: `Integer` Month strike occurred
4. **Incident Day**: `Integer` Date strike occurred
5. **Operator ID**: `String` Airline operator code
6. **Operator**: `String` Civil Aviation Organization Name
7. **Aircraft**: `String` Aircraft
8. **Aircraft Type**: `String` Type of aircraft
9. **Aircraft Make**: `String` International Civil Aviation Organization code for Aircraft Make
10. **Aircraft Model**: `String` International Civil Aviation Organization code for Aircraft Model
11. **Aircraft Mass**: `Integer` 1 = 2,250 kg or less: 2 = ,2251-5700 kg: 3 = 5,701-27,000 kg: 4 = 27,001-272,000 kg: 5 = above 272,000 kg
12. **Engine Make**: `String` Engine Make Code
13. **Engine Model**: `String` Engine Model Code
14. **Engines**: `Integer` Number of engines
15. **Engine Type**: `String` Type of power A = reciprocating engine (piston): B = Turbojet: C = Turboprop: D = Turbofan: E = None (glider): F = Turboshaft (helicopter): Y = Other
16. **Engine1 Position**: `String` Where engine # 1 is mounted on aircraft
17. **Engine2 Position**: `String` Where engine # 2 is mounted on aircraft
18. **Engine3 Position**: `String` Where engine # 3 is mounted on aircraft
19. **Engine4 Position**: `String` Where engine # 4 is mounted on aircraft
20. **Airport ID**: `String` International Civil Aviation Organization airport identifier for location of strike whether it was on or off airport
21. **Airport**: `String` Name of airport
22. **State**: `String` State
23. **FAA Region**: `String` FAA Region where airport is located
24. **Warning Issued**: `String` Pilot warned of birds/wildlife
25. **Flight Phase**: `String` Phase of flight during which strike occurred
26. **Visibility**: `String` Type of cloud cover, if any
27. **Precipitation**: `String` Precipitation
28. **Height**: `Integer` Feet Above Ground Level
29. **Speed**: `Float` Knots (indicated air speed)
30. **Distance**: `Float` Nautical miles from airport
31. **Species ID**: `String` International Civil Aviation Organization code for type of bird or other wildlife
32. **Species Name**: `String` Common name for bird or other wildlife
33. **Species Quantity**: `Integer`
34. **Flight Impact**: `String` Impact on the flight
35. **Fatalities**: `Integer` Number of human fatalities
36. **Injuries**: `Integer` Number of people injured
37. **Aircraft Damage**: `Integer`
38. **Radome Strike**: `Integer`
39. **Radome Damage**: `Integer`
40. **Windshield Strike**: `Integer`
41. **Windshield Damage**: `Integer`
42. **Nose Strike**: `Integer`
43. **Nose Damage**: `Integer`
44. **Engine1 Strike**: `Integer`
45. **Engine1 Damage**: `Integer`
46. **Engine2 Strike**: `Integer`
47. **Engine2 Damage**: `Integer`
48. **Engine3 Strike**: `Integer`
49. **Engine3 Damage**: `Integer`
50. **Engine4 Strike**: `Integer`
51. **Engine4 Damage**: `Integer`
52. **Engine Ingested**: `Integer`
53. **Propeller Strike**: `Integer`
54. **Propeller Damage**: `Integer`
55. **Wing or Rotor Strike**: `Integer`
56. **Wing or Rotor Damage**: `Integer`
57. **Fuselage Strike**: `Integer`
58. **Fuselage Damage**: `Integer`
59. **Landing Gear Strike**: `Integer`
60. **Landing Gear Damage**: `Integer`
61. **Tail Strike**: `Integer`
62. **Tail Damage**: `Integer`
63. **Lights Strike**: `Integer`
64. **Lights Damage**: `Integer`
65. **Other Strike**: `Integer`
66. **Other Damage**: `Integer`

# Sample values for the categorical features

1 Aircraft [B-757-200, DC-9, UNKNOWN, F-16, B-737-200, HA...]
2 Aircraft Type [A, nan, B, J]
3 Aircraft Make [148, 583, nan, 561, 443, 729, 395, 123, 70, 3...]
4 Aircraft Model [26, 90, nan, 13, 30, 24, 23, 14, 20, 3, 8, 2,...]
5 Engine Model [40, 10, nan, 1, 19, 37, 7, 4, 3, 34, 52, 31, ...]
6 Engine Type [D, nan, B, A, C, F, B/D, A/C, c, X]
7 Engine1 Position [1, 5, nan, 7, 4, 6, 2, 3, C]
8 Engine3 Position [nan, 1, 5, 4, CHANGE CODE, 3]
9 Airport ID [KCVG, PHLI, KJAX, KMCO, KJWN, KFSM, KMSY, KIK...]
10 State [KY, HI, FL, TN, AR, LA, MI, NJ, MN, nan, NY, ...]
11 FAA Region [ASO, AWP, ASW, AGL, AEA, nan, ACE, FGN, ANM, ...]
12 Warning Issued [nan, N, Y, n, y]
13 Flight Phase [CLIMB, TAKEOFF RUN, nan, LANDING ROLL, APPROA...]
14 Visibility [nan, DAY, NIGHT, DUSK, DAWN, UNKNOWN]
15 Precipitation [nan, NONE, FOG, RAIN, SNOW, FOG, RAIN, FOG, S...]
16 Species Quantity [1, 2-10, nan, 11-100, Over 100]
17 Flight Impact [nan, PRECAUTIONARY LANDING, OTHER, ABORTED TA..]
18 Operator ID [DAL, HAL, UNK, MIL, USA, BUS, SWA, PVT, UPS, ...]

# Outline what you plan to predict. How might this prediction be used in production or in practice?

We want to predict the type of species that will be injured based on the flight phase, the distance form airport, the state, the Visibility and others. This prediction can help Aviation build a system to block these animals based on the phases. Already Some airoprts use Bird Deterrent Systems like Predator calls (e.g., hawk or falcon cries), Distress calls of the birds themselves, Pyrotechnic sounds (like loud bangs). But they dont do much in mid flight because most birds dont flight in high altitude but on the take off and such they usually encounter birds and other animal.

We can further this study by monitoring animal (mostly bird movement) patterns and combining that data with the animal that our model predict and make precautions.

# APPROACH 1 Eden

# Process overview

At first we had select the columns by just the feel rather than the correlation. We had to redo everything after that. We also faced huge issue when training the model because the data set is very huge and hyper-parameter tuning using grid search was very hard.
Then we decide to do preprocessing ,remove corelated features and try bucketing and wothout bucketing too.

#EDA

# What are your X and Y variables?

Y variable is the newly engineered feature Bucketed species or higher categroy for the species. The column name is Bucket
X are the rest of the columns above.

# Classification or regression?

It is classifcation.

# How many observations?

We have 174104 samples which was reduced by doing preprocessing. We didnt have any Null species but we had 80771 species with UNK as their ID in other words these animals aren't identified hence are removed. 80771. Afterward we did SMOTE since our dataset is very imbalanced.

Before SMOTE but after clean up and train/test split it was (74602, 1072) and after SMOTE it is (188250, 1072). So we have 6 species each with 31375 entries in them.

# New feature engineered

Due to the amount of spicies typw a new feature was created by categorizing the species into a bigger umbrella. The created category/bucket are as follows:

    'SMALL BIRD': [
        'SPARROW', 'FINCH', 'WARBLER', 'WREN', 'VIREO', 'CHICKADEE', 'TITMOUSE','BUDGERIGAR',
        'SWIFT', 'HUMMINGBIRD', 'NUTHATCH', 'WAXWING', 'JUNCO', 'BUNTING',
        'KINGLET', 'BUSHTIT', 'PIPIT', 'VERDIN', 'GNATCATCHER', 'TOWHEE',
        'PHOEBE', 'VIREOS', 'MYNA', 'BULBUL',  'PINE SISKIN',
        'LONGSPUR', 'CHAT', 'UNKNOWN SMALL BIRD',  'MUNIA', 'SWALLOW',
        'WAXBILL', 'TANAGER','CANARY', 'DRONGO', 'WHITE-EYE', 'ELAENIA',
        'HORNED LARK', 'ROSE-BREASTED GROSBEAK', 'NORTHERN PARULA', 'AMERICAN REDSTART',
        'VEERY', 'EURASIAN SKYLARK', 'BOBOLINK', 'DICKCISSEL', 'RED AVADAVAT',
        'CHUCK-WILL\'S-WIDOW', 'LARKS', 'RED-NAPED SAPSUCKER', 'DOWNY WOODPECKER',
        'HAIRY WOODPECKER', 'RED-BREASTED SAPSUCKER', 'COMMON YELLOWTHROAT',
        'WHITE-WINGED CROSSBILL', 'PINE GROSBEAK', 'EVENING GROSBEAK',
        'BLACK-HEADED GROSBEAK', 'BLUE GROSBEAK', 'RED CROSSBILL', 'REDWING',
        'COMMON REDPOLL', 'HOARY REDPOLL', 'ROADRUNNER'
    ],
    'MEDIUM BIRD':[
        'UNKNOWN MEDIUM BIRD','DOVE', 'CUCKOO', 'SHRIKE', 'KINGBIRD', 'MOCKINGBIRD','RED-LEGGED PARTRIDGE',
        'THRASHER', 'JAY', 'MARTIN', 'ROBIN', 'THRUSH', 'CARDINAL', 'COWBIRD', 'ORIOLE', 'CROW', 'RAVEN', 'JAY', 'MAGPIE', 'STARLING', 'ROOK', 'GRACKLE',
        'BLACKBIRD', 'BOAT-TAILED', 'RED-WINGED','WHIP-POOR-WILL', 'BELTED KINGFISHER', 'YELLOW-BELLIED SAPSUCKER',
        'RED-BELLIED WOODPECKER', 'WOODPECKERS, PICULETS', 'WOODPECKERS',
        'RED-HEADED WOODPECKER', 'MONK PARAKEET', 'NANDAY PARAKEET',
        'OLIVE-THROATED PARAKEET', 'PARROTS', 'GREAT KISKADEE', 'SCALED QUAIL',
        'CHUKAR', 'QUAILS', 'EURASIAN THICK-KNEE', 'DOUBLE-STRIPED THICK-KNEE',
        'COMMON PAURAQUE', 'GREATER ROADRUNNER'
    ],
    'LARGE BIRD': [
        'GOOSE', 'GEESE', 'DUCK', 'HERON', 'EGRET', 'CORMORANT', 'PELICAN', 'NORTHERN FULMAR',
        'CRANE', 'SWAN', 'TURKEY', 'IBIS', 'LOON', 'RAIL', 'GALLINULE',
        'BITTERN', 'PHALAROPE', 'MOORHEN', 'AVOCET', 'GODWIT', 'CURLEW',
        'SPOONBILL', 'JAEGER', 'SHEARWATER', 'ALBATROSS', 'STORK', 'PETREL',
         'PHEASANT', 'GROUSE', 'GUINEAFOWL', 'FRANCOLIN', 'PTARMIGAN','UNKNOWN LARGE BIRD'
        'PARTRIDGE', 'TURKEY','DOWITCHER', 'DUNLIN',
        'SKIMMER', 'TROPICBIRD', 'NODDY', 'FRIGATEBIRD',
        'SHOREBIRD','SANDPIPER', 'PLOVER', 'STILT', 'OYSTERCATCHER',
        'PHALAROPE', 'TERN', 'GULL', 'SKUA', 'KITTIWAKE',
        'MURRE', 'GUILLEMOT', 'PUFFIN', 'ALBATROSS', 'SHEARWATER',
        'PETREL', 'GREBE', 'COOT', 'RAIL', 'GALLINULE', 'WATERTHRUSH',
        'YELLOWLEGS', 'SORA', 'MERGANSER', 'TEAL', 'DUCK', 'WIGEON',
        'SHOVELER', 'EIDER', 'SNIPE', 'TURNSTONE', 'AVOCET', 'MOORHEN',
        'WOODCOCK', 'SCOTER', 'GADWALL', 'PINTAIL',
        'EAGLE', 'FALCON', 'OSPREY', 'VULTURE', 'CARACARA', 'KITE',
        'MERLIN', 'KESTREL', 'BUZZARD', 'OWL', 'EAGLE', 'HAWK',  'HARRIER', 'OSPREY', 'CARACARA',
        'KITE', 'SCREECH', 'NIGHTJAR', 'NIGHTHAWK', 'POORWILL', 'SAW-WHET','MALLARD', 'ANHINGA', 'NORTHERN FLICKER', 'BRANT', 'NORTHERN LAPWING',
        'GRAY PARTRIDGE', 'NORTHERN BOBWHITE', 'LESSER SCAUP', 'CANVASBACK',
        'WHIMBREL', 'GREATER SCAUP', 'WILLET', 'SANDERLING', 'BUFFLEHEAD',
        'REDHEAD', 'COMMON GOLDENEYE', 'BARROW\'S GOLDENEYE', 'RED-FOOTED BOOBY',
        'RED KNOT', 'SOUTHERN LAPWING', 'YELLOW-HEADED CARCARA'
    ],
    'BIRD':['UNKNOWN BIRD OR BAT','BIRD', 'UNKNOWN BIRD'],
    'MAMMAL':['UNKNOWN TERRESTRIAL MAMMAL',
        'DEER', 'MOOSE', 'ELK', 'HORSE', 'COW', 'PIG', 'SWINE', 'BURRO',
        'PRONGHORN', 'BISON', 'ANTELOPE','CARIBOU',  'PECCARY',
         'HARE', 'SKUNK', 'RACCOON', 'OPOSSUM', 'WOODCHUCK', 'BAT',
        'SQUIRREL', 'MARMOT', 'MUSKRAT', 'BEAVER', 'ARMADILLO', 'RODENT',
        'MINK', 'WEASEL', 'FERRET','RAT','SHREW', 'MOUSE',  'VOLE', 'CHIPMUNK',
        'PORCUPINE', 'RABBIT', 'COTTONTAIL',
        'FOX', 'COYOTE', 'WOLF', 'DOG', 'CAT', 'BOBCAT', 'LYNX', 'BEAR',
        'CANID', 'FELINE', 'RINGTAIL', 'JACKAL', 'MOUNTAIN LION', 'COATI', 'OTTER','BADGER', 'YUMA MYOTIS', 'LONG-EARED MYOTIS', 'COMMON PIPISTRELLE','MYOTIS',
        'LONG-LEGGED MYOTIS'
    ],
    'REPTILE & AMPHIBIAN': [
        'TURTLE', 'SNAKE', 'LIZARD', 'ALLIGATOR', 'CROCODILE', 'FROG',
        'TOAD', 'IGUANA', 'TERRAPIN', 'GECKO', 'TORTOISE',  'COOTER','SLIDER', 'MOCCASIN'
    ],
    'UNKNOWN / OTHER': [
        'UNKNOWN',
         'OTHER', 'UNIDENTIFIED','NAN'
    ]

# What is the distribution of each feature? You don't need to show every feature's distribution. Focus on features which are imbalanced or may present a problem when training a model.

![alt text](Species-Dist.png)
![alt text](distribution_by_flight_phase.png)
![alt text](Distribution_per_feature.png)
To balance we did SMOTE. Before SMOTE it was (74602, 1072) and after SMOTE it is (188250, 1072).
So we have 6 category/bucket each with 31375 entries in them.

# Correlation - are some features strongly correlated?

To see the corelation of the features we have used cramers for categorical and correlation matrix for the numeric features. There was some corelation and due to that we have removed the features listed below.

['Speed', 'Aircraft Make','Operator','Engine3 Damage','Incident Year','Engine2 Damage','Lights Damage', 'Engine1 Damage', 'State', 'Wing or Rotor Damage','Engines', 'Engine Ingested','Engine Model','Distance','Engine1 Position','Airport','Engine4 Position','Tail Damage','Fatalities', 'Aircraft Type','FAA Region', 'Species Name','Engine Type','Aircraft Model','Windshield Damage', 'Engine3 Position','Airport ID','Species ID'] are highly correlated.

![alt text](merged_correlation.png)

# Feature importance.

Are you using all features for X. If so, make a case for that. If not, make a case for the feature selection you used.
Feature engineering

# Which features needed feature engineering?

All these needed Imputation 'Aircraft Type', 'Aircraft Make', 'Aircraft Model', 'Aircraft Mass',
'Engine Make', 'Engine Model', 'Engines', 'Engine Type',
'Engine1 Position', 'Engine2 Position', 'Engine3 Position',
'Engine4 Position', 'Airport', 'State', 'FAA Region', 'Warning Issued',
'Flight Phase', 'Visibility', 'Precipitation', 'Height', 'Speed',
'Distance', 'Species Name', 'Species Quantity', 'Flight Impact',
'Fatalities', 'Injuries'

# Label encoding vs. one hot encoding

We used one hot encoding becuase none of the categorical variable had any ordinal information hence we didnt want the data set to have such kind of order hence why we used One hot encoding.

# Cross features

# More advanced encoding / feature engineering you might have completed.

At first I did the target encoding but then due to the comment added on the project 2 I have changed it to one-hot encoder.

# Model fitting

# Train / test splitting

Our training consists 80% of our data and is startified by our Y(The bucketed feature)

# Does your dataset have a risk of data leakage? Describe those risks.

Our dataset doesnt have any data leakage risk. Everyhting was done after spliting. We dont have any time series related work.

# Many have tried multiple model types - what is the thought process for trying these types?

We have tried Logistic, Random forest(Before and after PCA), SVC, Random forest, neural network.
The reason that these models were choose was because first they all work for classification mode. The logistic had a very low accuracy and there needed to be a more complex mod

# Which model did you select, why?

# What was the process for hyper parameter selection if applicable.

Due to the amount of the entries and the limit of resources we were only able to implement the grid search just for the SVC with only 5% of the dataset. For other models we set our own values for the hyper-parameters.

# Validation / metrics

# Which metrics did you weigh most heavily, why?

Since our model is a classfication we are using classfication metrics. We used Recall because our bucket had a very imbalanced data.

# Confusion matrix and confusion discussion for classification problems. Results, weakness for each model

Logistics
![alt text](LogisticRegression_confusion.png)
![alt text](LogisticRegression_metrics.png)
Random Forest (Before PCA)
![alt text](RandomForestClassifier_confusion.png)
![alt text](RandomForestClassifier_metrics.png)
SVC

Random Forest (After doing PCA)
![alt text](RandomForestClassifierwithPCA_confusion.png)
![alt text](RandomForestClassifierwithPCA_metrics.png)

Neural network
![alt text](NeuralNetworks_confusion.png)
![alt text](NeuralNetworks_metrics.png)

                             | Accuracy | Accuracy    |             |          |          |

| Model                      | Train    | Test     | Overfitting | Metric 5 | Weakness |
| -------------------------- | -------- | -------- | ----------- | -------- | -------- |
| Logistics(No PCA)          | Value 2  | Value 3  | Value 4     | Value 5  |
| SVC                        | Value 7  | Value 8  | Value 9     | Value 10 |
| Random Forest (Before PCA) | Value 12 | Value 13 | Value 14    | Value 15 |
| Random Forest (After PCA)  | Value 17 | Value 18 | Value 19    | Value 20 |
| Neural network             | Value 22 | Value 23 | Value 24    | Value 25 |

# Give 2-4 prediction examples from your data.

# Give 2 prediction examples which are new or synthesized.

# Identify if model is overfitting or underfitting

Our random forest was overfitting before we add PCA. We wanted 95% of the variance and that is how we calculated the PCA.

# Identify and apply techniques to mitigate underfitting/overfitting

# Production

# Give any advice for deployment of use of this model

All preprocessing are dummped in the .pkl format so anyone who wants to integrate this should apply these transformation before using the model to make prediction. As a sample you can run predict.py file to see how it works.

# Outline precautions about its use here

# Going further

If there is a chance to use a better machine to compute this model, I would prefer to decrease the granularity right now we have added any species to huge categories which is not that much usefull because what is the point if the model suggest huge bird , medium bird. I

# What could make this model better?

Hyperparameter tuning, removing bucketing and balancing the data by collecting more entries, If bucketing is going to be used then be more specific on the categories.

# APPROACH 2 Vimbai

# Feature importance. - Are you using all features for X? If so, make a case for that. If not, make a case for the feature selection you used.

NO, it wasn’t necessary to use all of them for our target variable. Features post-impact are not relevant in our pre-incident predictive study. So, I excluded everything post-impact and the damages as well since our primary focus was on the species group. We did perform a PCA but overall, background and non-statistical knowledge was important.

Features used: 'Species Group', 'Incident Month', 'Flight Phase Cleaned','Aircraft Mass', 'Engine Type Cleaned', 'State', 'Height', 'Speed'

# Feature engineering

# Which features needed feature engineering? Discussion:

Looking at numerical features: Height, aircraft mass and speed underwent scaling and had their missing values handled via median imputation. Then regarding categorical variables they were bucketed: Flight Phase, State, engine type and Species Name was consolidated into broader groups.

# Label encoding vs. one hot encoding?

All catergorical variables were encoded by One-Hot Encoding.

# Cross features? More advanced encoding / feature engineering you might have completed.

We did not do any cross features but after going through the models maybe height and speed as they had the highest influence on predicting the target variables. The image below has an example of the

# Model fitting

# Train / test splitting - How was this performed? How did you decide the train/test sizes?

I did the standrad 80% training data and 20% testing split, stratisfied by Species Group. The dataset size was sufficient enough to do this.

# Does your dataset have a risk of data leakage? Describe those risks.

YES, it does. If we had decided to focus on time like seasons and then had to order it. Another major one was using scale and mean imputations, I used SimpleImputer() before splitting but that is why we only fitted it on the training data ONLY.

# Which model did you select, why?

Despite having tried 4 models, we went with Random Forest at the end due to its high recall (more important for airplane companies), auc score and other metrics.
![alt text](image-6.png)

# Many have tried multiple model types - what is the thought process for trying these types?

We tried Log Reg at first as a baseline to see the data's linear relationships and other factors. Then we proceed to Linear SVC, worked better for the large dataset with linear patterns but theere was still an issue with distaigusihing betweeb classes especially for known small birds and other smaller groups that made the data heavily imbalanced and why we resorted to SMOTE and decreased granularity in categorical bucketing earlier. From then we jumped a decsion tree to see if Random Forest would do any better and it did, in all aspects and that's why its our frontrunner. For hyper-parameter tuning for optimization, grid search was used.

# Only use models learned in class (linear regression, logistic regression, SVC/SVM, decision trees (including random forests, etc))

# What was the process for hyper parameter selection if applicable?

For hyper-parameter tuning for optimization, grid search was used for log and svc.

# Validation / metrics

# Which metrics did you weigh most heavily, why? - Accuracy, r^2, balanced accuracy, ROC, AUC, Recall, Precision, etc..

We used RECALL (Macro Avg) because our class is very imbalanced. AUC for model performances and log-loss helped in measuring how well each model's probabilty outputs matched with real-life. We also had accurancy, precison and others metrics obtrained from the classifiaction report.

Looking at Macro AUC and micro AUC:
In first place we have Random Forest that handles class imbalance while maintaining strong overall performance across all classes. This model strikes the best balance between ranking ability and class-wise discrimination while in runner up is the decision tree that outperforms Logistic Regression; it captures non-linbeararity better by the looks of it. And last is log regression with the lowest macro scores and its mostly due to its struggle to differenraite between classes.
![alt text](image-7.png)

# Confusion matrix and confusion discussion for classification problems

![alt text](image-3.png)
![alt text](image-4.png)
Additionally we used confusion matrixes too. Looking at the different matrices across the different models, there are a few clear takeaways. Random Forest does the best job handling UNKNOWN classes, with high true positive counts for UNKNOWN MEDIUM BIRD and UNKNOWN SMALL BIRD. In contrast, Logistic Regression and SVC struggles significantly with these, often scattering predictions across multiple classes. Rare animals are another challenge—every model misclassifies them frequently, especially into categories like Gulls and Pigeons. This likely points to class imbalance or overlapping features, which might need additional feature engineering or resampling.

Logistic Reg showed the most class confusion overall, with predictions spread too broadly and a lack of clear separations—especially for Gulls & Water Birds. Decision, though better, still overfits and mispredicts them in huge numbers. Random Forest helps mitigate this issue by capturing non-linear patterns more effectively, making it the strongest choice for handling these types of classifications.

One trend across all the models is the high confusion between visually similar bird types, like Swallows & Swifts, Small Land Birds, and Raptors. These misclassifications make sense given real-world similarities in size, flight patterns, and habitat. If interpretability is a priority, grouping certain species or extracting more distinguishing features could improve accuracy. While Random Forest comes out better, refining class definitions and balancing dataset representation will be key for future improvements.

# Give 2 prediction examples which are new or synthesized.

The overall concensus is each of these poorly performs on rarer classes and may over-relies on height and scv on speed. So more weight on numerical features than contextual ones.

# Overfitting/Underfitting

# 1. Identify if model is overfitting or underfitting

# 2. Identify and apply techniques to mitigate underfitting/overfitting

The underfitting on minority classes was cleared by SMOTE and/or "weight' = "balanced".

# Production

# 1. Give any advice for deployment of use of this model --> Outline precautions about its use here and Going further.

If there are new bird species or major aircraft design changes that are not in training, this will cause a problem so this must be monitored.

# What could make this model better? More data, additional features, more augmentation etc.

Some kind of Boosting like Gradient Boost would be helpful here and for data, more records of weather conditions wouldve helped. I think knowing bird migration seasons and patterns would help this a lot since State did have some statistical influence.
![alt text](image-5.png)
