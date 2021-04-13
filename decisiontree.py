import sys, json
import pandas as pd
import utility as util

"""
Implementation of ID3 Decison-Tree Algorithm """
# -------------------------------------------- #

class DecisionTreeID3:
    def __init__(self):
        self.tree = dict()


    def formulate_tree(self, dataset):
        print(dataset.head(15), "\n")

        target_col = dataset.iloc[:, -1]
        choices = list(set(target_col))
        
        attr_index, max_ig = util.max_information_gain_attr_index(dataset)
        print("\nMax IG: %.5f" % (max_ig), "\n")

        childs = util.get_childs(dataset, attr_index)
        self.tree[dataset.columns[attr_index]] = childs

        impure_childs = []

        for key, value in childs.items():
            if value[0] == 0:
                self.tree[key] = choices[1]
            elif value[1] == 0:
                self.tree[key] = choices[0]
            elif value[0] != 0 or value[1] != 0:
                impure_childs.append(key)

        for x in impure_childs:
            sub_dataset = util.reduce_dataset(
                dataset, attr_index, dataset.columns[attr_index], x
            )

            self.formulate_tree(sub_dataset)


    def display(self):
        print("Result:", json.dumps(self.tree, indent = 4))

"""
main function """
# ------------- #

def main():

    argc = len(sys.argv)

    if argc == 2:
        dataset_filename = sys.argv[1]
        df = None

        try:
            df = pd.read_csv(dataset_filename, sep="\t")
        except:
            raise FileNotFoundError(
                "The file '%s' does not exist.\n" % (dataset_filename)
                )

        tree_obj = DecisionTreeID3()
        tree_obj.formulate_tree(df)
        tree_obj.display()
        
    else:
        message = "Error!\n$python <python-file> <dataset-filename>"
        raise Exception(message)

"""
driver code """
# ----------- #

if __name__ == "__main__":
    main()
