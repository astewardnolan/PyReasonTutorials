import networkx as nx
import pyreason as pr
from pprint import pprint
import numba 
import matplotlib.pyplot as plt
import numpy as np


# Customer Data
people = ['John', 'Mary', 'Justin', 'Alice', 'Bob', 'Eva', 'Mike']
people_details = [
    ('John', 'M', 'New York', 'NY'),
    ('Mary', 'F', 'Los Angeles', 'CA'),
    ('Justin', 'M', 'Chicago', 'IL'),
    ('Alice', 'F', 'Houston', 'TX'),
    ('Bob', 'M', 'Phoenix', 'AZ'),
    ('Eva', 'F', 'San Diego', 'CA'),
    ('Mike', 'M', 'Dallas', 'TX')
]

# Creating a dictionary of peoples with their details
people_dict = {i: people for i, people in enumerate(people_details)}

# Pet Data
pet_details = [
    ('Dog', 'Mammal'),
    ('Cat', 'Mammal'),
    ('Rabbit', 'Mammal'),
    ('Parrot', 'Bird'),
    ('Fish', 'Fish')
]

# Creating a dictionary of pets with their details
pet_dict = {i: pet for i, pet in enumerate(pet_details)}

# Pet Ownerships (who owns which pet)
pet_ownerships = [('person_1', 'Pet_1'), ('person_2', 'Pet_1'), ('person_2', 'Pet_0'), ('person_0', 'Pet_0'),
                  ('person_3', 'Pet_2'), ('person_4', 'Pet_2'), ('person_5', 'Pet_3'), ('person_6', 'Pet_4'),
                  ('person_0', 'Pet_4'),('person_4', 'Pet_4')]

# Friendships (who is friends with whom)
friendships = [('person_2', 'person_1'), ('person_0', 'person_1'), ('person_0', 'person_2'),
               ('person_3', 'person_4'), ('person_4', 'person_0'), ('person_6', 'person_0'), 
               ('person_5', 'person_6'), ('person_4', 'person_2')]

# Create a directed graph
g = nx.DiGraph()




# Add nodes for peoples
for people_id, details in people_dict.items():
    attributes = {
        'c_id': people_id,
        'name': details[0],
        'gender': details[1],
        'city': details[2],
        'state': details[3],
    }
    name = "person_" + str(people_id)
    g.add_node(name, **attributes)

# Add nodes for pets
for pet_id, details in pet_dict.items():
    dynamic_attribute = f"Pet_{pet_id}"
    attributes = {
        'pet_id': pet_id,
        'species': details[0],
        'class': details[1],
        dynamic_attribute: 1
    }
    name = "Pet_" + str(pet_id)
    g.add_node(name, **attributes)



# Add edges for relationships
for f1, f2 in friendships:
    g.add_edge(f1, f2, Friends=1)
print(g)
for owner, pet in pet_ownerships:
    g.add_edge(owner, pet, owns_pet=1)



# Load the graph into PyReason
pr.load_graph(g)

# Set PyReason settings
pr.settings.verbose = True
pr.settings.atom_trace = True



@numba.njit
def potential_customer_ann_fn(annotations, weights):
    # num_friends is the number of friends a person has
    num_friends = len(annotations[1])
            
    if num_friends >= 2:
        lower_bound = 0.5
        upper_bound = 1
    else:
        lower_bound = 0 
        upper_bound = 0  
    
    return lower_bound, upper_bound



# Add the annotation function to PyReason
pr.add_annotation_function(potential_customer_ann_fn)


# Define logical rules

# Potential customer detection: a customer is a potential customer if they are friends with a current customer.
pr.add_rule(pr.Rule('potential_customer(x) : potential_customer_ann_fn <- customer(y), Friends(x,y)', 'potential_customer_rule'))

# Target customer detection: a potential customer becomes a target customer if their potential customer score is between 0.5 and 1,
# and they share a pet with a friend.
pr.add_rule(pr.Rule('target_customer(x) <- potential_customer(x) : [0.5, 1], Friends(x,y), owns_pet(y, p), owns_pet(x, p)', 'target_customer_rule'))

# Adding a fact that sets person_0 as a customer at timestep 0, person_2 as a customer at timestep 1
pr.add_fact(pr.Fact('customer(person_0)', 'customer_fact', 0, 1))
pr.add_fact(pr.Fact('customer(person_2)', 'customer_fact', 1, 1))


# Perform reasoning over 1 timestep
interpretation = pr.reason(timesteps=1)


# Display the interpretation
interpretations_dict = interpretation.get_dict()
pprint(interpretations_dict)


#Filter and sort nodes based on specific attributes
df1 = pr.filter_and_sort_nodes(interpretation, ['potential_customer', 'target_customer'])

#Display filtered node and edge data
for t, df in enumerate(df1):
    print(f'TIMESTEP - {t}')
    print(df)
    print()


pr.save_rule_trace(interpretation)
