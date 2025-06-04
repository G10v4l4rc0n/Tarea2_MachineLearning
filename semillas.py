import re
import pandas as pd

######IMPORTACIÓN Y TRANSFORMACIÓN DE DATASET A DATAFRAME
#IMPORTACIÓN
with open("seeds_dataset.txt", 'r') as file:
    data_list = re.split('\t|\n', file.read())

#TRANSFORMACIÓN A LISTA
copy = data_list.copy()
for value in copy:
    try:
        index_to_remove = copy.index('')
    except:
        break
    else:
        copy.remove('')

#TRANSFORMACIÓN A DATAFRAME
final_list, sub_list = [], []
cont = 0;
for i in range(210):
    final_list.append(copy[8*i:8*(i+1)])

#check if new list has the right values
for val in final_list[:5]:
    print(val)

data ={
    "area": [],
    "perimeter": [],
    "compactness": [],
    "length of kernel": [],
    "width of kernel": [],
    "asymmetry coefficient": [],
    "length of kernel groove": [],
    "classification": []
}
for sub_list in final_list:
    data["area"].append(sub_list[0])
    data["perimeter"].append(sub_list[1])
    data["compactness"].append(sub_list[2])
    data["length of kernel"].append(sub_list[3])
    data["width of kernel"].append(sub_list[4])
    data["asymmetry coefficient"].append(sub_list[5])
    data["length of kernel groove"].append(sub_list[6])
    data["classification"].append(sub_list[7])
df = pd.DataFrame(data)
print(df)


#

























