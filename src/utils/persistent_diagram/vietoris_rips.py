from ...utils.persistent_diagram import PersistenceDiagrams


def vietoris_rips_persistence_diagram(points, max_edge_length, tool='ghudi'):
    if tool == 'ghudi':
        import gudhi
        rips_complex = gudhi.RipsComplex(points=points,
                                         max_edge_length=max_edge_length)
    
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
        return PersistenceDiagrams.from_list([
            [pt for d, pt in diag if d == 0],
            [pt for d, pt in diag if d == 1]
        ])
    else:
        raise ValueError