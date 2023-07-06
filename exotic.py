def compute_all_intersections(hyperplanes):
    """ Computes all possible intersections of Hyperplane objects. 
    
        In the worst case this algorithm has O(2^n) time complexity, where n is 
        the length of `hyperplanes`. This occurs when every possible intersection 
        the hyperplanes is unique. In practice this situation is rare: note that 
        HJ = J or HJ = H or dim(HJ) < min(dim(H), dim(J))for hyperplanes H and J. 
        This algorithm is written to considerably improve the running time when 
        many intersections are the same, and to avoid computing any intersection 
        more than once. 

        Args:
            hyperplanes: a list of hyperplanes.

        Returns:
            A set of all possible non-empty intersections, including the elements from 
            `hyperplanes`
    """

    # the final collection of all intersections
    out = set(hyperplanes)

    # we store the spaces generated on the last iteration along with the index of the 
    # space in hyperplanes which was intersected to generate it on the last iteration
    last_inters = {c : i for i, c in enumerate(hyperplanes)}
    while last_inters:
        
        # intersections being generated in this iteration
        new_inters = dict()
        for c, last_i in last_inters.items():

            # intersecting a point space with anything else can only ever yield 
            # itself or the empty set, so no point doing intersections
            if c.is_point():
                continue
            
            # intersections are tried in the order given by irr_components, so 
            # we start generating new intersections from last_i + 1
            for j, d in enumerate(hyperplanes[last_i + 1:]):

                new = c.intersect(d)

                # skip empty intersections and spaces generated in previous iterations
                if not new.is_empty() and new not in out:
                    
                    # new has already been generated in this iteration
                    if new in new_inters:
                        # we need to take the minimum to avoid missing an intersection 
                        # eg if AB = AD, it may be that ABC is a new set
                        new_inters[new] = min(last_i + j + 1, new_inters[new])

                    else:
                        new_inters[new] = last_i + j + 1

        out.update(new_inters.keys())
        last_inters = new_inters

    return out
