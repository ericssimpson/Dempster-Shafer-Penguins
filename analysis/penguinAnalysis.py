#Import analysis methods:
from dempsterShafer import *
import dempsterShafer as ds
import json

#Import dataset for analysis:
penguins = pd.read_csv("analysis/penguins.csv")
#?print(dataset.head())


#Dataset diagnostics:
def view_diagnostics(dataset):
    fd, axes = plt.subplots(2, 2, figsize=(12, 10))
    for fd, ax in [('culmen_length_mm',axes[0,0]), ('culmen_depth_mm',axes[0,1]), ('flipper_length_mm',axes[1,0]), ('body_mass_g',axes[1,1])]:
        sns.histplot(dataset, x=fd, hue='species', element='step', ax=ax) 
    sns.jointplot(
        data=dataset,
        x="culmen_length_mm", y="culmen_depth_mm", hue="species",
        kind="kde"
    )
    plt.show()
    return None
#?view_diagnostics(dataset)


#Cleaning dataset:
penguins = penguins.dropna()
bad_indexs = penguins[ (penguins['sex'] != 'MALE') & (penguins['sex'] != 'FEMALE')].index
penguins.drop(bad_indexs, inplace = True)
#?penguins.to_csv('penguins_cleaned.csv')


#Splitting dataset by sex:
penguins_male = penguins[penguins['sex'] == 'MALE']
#?view_diagnostics(penguins_male)

penguins_female = penguins[penguins['sex'] == 'FEMALE']
#?view_diagnostics(penguins_female)


#Create prediction:
def prediction(dataset, class_range):
    correct = 0
    incorrect = 0
    indeterminate = 0
    for index, row in dataset.iterrows():

        sample = row
        masses = {}

        for f in fields:

            hypothesis_count = powerset(dataset.species.unique())
            h = hypothesis(dataset.species, class_range, f, row[f])
            hypothesis_counts(hypothesis_count, h)
            total = sum([i[1] for i in hypothesis_count])

            for i in range(0, len(hypothesis_count)):

                hypothesis_count[i][1] = hypothesis_count[i][1]/total
                masses[f] = hypothesis_count
        
        masses_comb1 = combine_masses(masses['culmen_length_mm'], masses['culmen_depth_mm'])
        masses_comb2 = combine_masses(masses_comb1, masses['flipper_length_mm'])
        masses_comb3 = combine_masses(masses_comb2, masses['body_mass_g'])
        
        
        ## Find the highest output of combined Mass
        vals = 0

        for i in range(0, len(masses_comb3)):

            if masses_comb3[i][1] > vals:

                most_likely = masses_comb3[i][0]
                vals = masses_comb3[i][1]

        if len(most_likely) == 1:

            if most_likely[0] == row["species"]:
                correct += 1
            
            else:
                incorrect += 1

        ## For situations where multiple classes are selected as the most likely
        ## We declare them wrong.

        ## What could we do here to provide a second way of differentiating between the two classes? 
        ## Try something.
        if len(most_likely) != 1:
            indeterminate += 1

        ## For each observation; we should also be able to get a sense of our belief and the plausibility.
        output1 = get_output(masses_comb1)
        output2 = get_output(masses_comb2)
        output3 = get_output(masses_comb3)

    return_dict = {'correct': correct, 'incorrect': incorrect, 'indeterminate': indeterminate}
    return return_dict


#Generate prediction:
total_correct = 0
total_incorrect = 0
total_indeterminate = 0

fields =  ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

class_range_male = class_range_output(penguins_male, penguins_male.species, fields)
prediction_male = prediction(penguins_male, class_range_male)

class_range_female = class_range_output(penguins_female, penguins_female.species, fields)
prediction_female = prediction(penguins_female, class_range_female)

total_correct += prediction_male['correct']
total_correct += prediction_female['correct']

total_incorrect += prediction_male['incorrect']
total_incorrect += prediction_female['incorrect']

total_indeterminate += prediction_male['indeterminate']
total_indeterminate += prediction_female['indeterminate']


#The final output:
print(f"We had {total_correct} correct classifications")
print(f"We had {total_incorrect} incorrect classifications")
print(f"We had {total_indeterminate} indeterminate classifications")