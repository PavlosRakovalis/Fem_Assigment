import numpy as np

def rotation_matrix(coords1, coords2):
    """
    Υπολογίζει το μητρώο μετασχηματισμού Τ για στοιχείο ράβδου
    σε 2D ή 3D, με βάση τις συντεταγμένες των δύο κόμβων.

    Είσοδος:
        coords1, coords2 : π.χ. [x1, y1] ή [x1, y1, z1]
    Έξοδος:
        T : πίνακας μετασχηματισμού (2x4 για 2D ή 2x6 για 3D)
    """
    #v = np.array(coords2) - np.array(coords1)
    #L = np.linalg.norm(v)
    #lmn = v / L  # συνήμιτονα κατεύθυνσης

    x1, y1, z1 = coords1
    x2, y2, z2 = coords2

    L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    l = (x2 - x1) / L
    m = (y2 - y1) / L
    n = (z2 - z1) / L

    if len(coords1) == 2:
        #l, m = lmn
        T = np.array([
            [ l,  m,  0,  0],
            [ 0,  0,  l,  m]
        ])
    elif len(coords2) == 3:
        #l, m, n = lmn
        T = np.array([
            [l, m, n, 0, 0, 0],
            [0, 0, 0, l, m, n]
        ])
    else:
        raise ValueError("Οι συντεταγμένες πρέπει να είναι 2D ή 3D.")

    return T, L

# Παράδειγμα χρήσης σε λούπα
nodes = [
    ([0, 0, 0], [1, 0, 0]),
    ([0, 0, 0], [0, 1, 0]),
    ([0, 0, 0], [1, 1, 0]),
    ([0, 0, 0], [1, 1, 1])
]

for n1, n2 in nodes:
    T, L = rotation_matrix(n1, n2)
    print(f"Κόμβοι: {n1} -> {n2}, Μήκος = {L:.3f}")
    print(f"T =\n{T}\n")
    print(T[0][0])


    def rotation_matrix_from_element(element_num, elements_df, nodes_df):
        """
        Υπολογίζει το μητρώο μετασχηματισμού Τ για ένα στοιχείο με βάση τον αριθμό του.
        
        Είσοδος:
            element_num: Ο αριθμός του στοιχείου
            elements_df: DataFrame με στήλες ['Element Number', 'Node1', 'Node2', 'E', 'V']
            nodes_df: DataFrame με στήλες ['Node', 'X', 'Y', 'Z']
        
        Έξοδος:
            T: Πίνακας μετασχηματισμού
            L: Μήκος στοιχείου
        """
        # Βρες τη γραμμή του στοιχείου
        element_row = elements_df[elements_df['Element Number'] == element_num].iloc[0]
        
        node1_id = element_row['Node1']
        node2_id = element_row['Node2']
        
        # Βρες τις συντεταγμένες των κόμβων
        coords1 = nodes_df[nodes_df['Node'] == node1_id][['X', 'Y', 'Z']].values[0]
        coords2 = nodes_df[nodes_df['Node'] == node2_id][['X', 'Y', 'Z']].values[0]
        
        # Υπολογισμός μήκους και συντελεστών κατεύθυνσης
        x1, y1, z1 = coords1
        x2, y2, z2 = coords2
        
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        
        l = (x2 - x1) / L
        m = (y2 - y1) / L
        n = (z2 - z1) / L
        
        # Δημιουργία πίνακα μετασχηματισμού
        T = np.array([
            [l, m, n, 0, 0, 0],
            [0, 0, 0, l, m, n]
        ])
        
        return T, L

    # Παράδειγμα χρήσης
    # T, L = rotation_matrix_from_element(1, elements, nodes)
    # print(f"Στοιχείο 1: Μήκος = {L:.3f}")
    # print(f"T =\n{T}")