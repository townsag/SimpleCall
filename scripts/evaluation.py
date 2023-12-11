import parasail
import re

BASES = ['A', 'C', 'G', 'T']

# parasail alignment configuration
ALIGNMENT_GAP_OPEN_PENALTY = 8
ALIGNMENT_GAP_EXTEND_PENALTY = 4
ALIGNMENT_MATCH_SCORE = 2
ALIGNMENT_MISSMATCH_SCORE = 1
GLOBAL_ALIGN_FUNCTION = parasail.nw_trace_striped_32
LOCAL_ALIGN_FUNCTION = parasail.sw_trace_striped_32
MATRIX = parasail.matrix_create("".join(BASES), ALIGNMENT_MATCH_SCORE, -ALIGNMENT_MISSMATCH_SCORE)

def elongate_cigar(short_cigar):
    cigar_counts = re.split('H|X|=|I|D|N|S|P|M', short_cigar)
    cigar_strs = re.split('[0-9]', short_cigar)
    
    cigar_counts = [c for c in cigar_counts if c != '']
    cigar_strs = [c for c in cigar_strs if c != '']
    
    assert len(cigar_strs) == len(cigar_counts)
    
    longcigar = ''
    for c, s in zip(cigar_counts, cigar_strs):
        longcigar += s*int(c)
    return longcigar, cigar_counts, cigar_strs


def alignment_accuracy(y, p, alignment_function = GLOBAL_ALIGN_FUNCTION, matrix = MATRIX, 
                       open_penalty = ALIGNMENT_GAP_OPEN_PENALTY, extend_penalty = ALIGNMENT_GAP_EXTEND_PENALTY):
    """Calculates the accuracy between two sequences
    Accuracy is calculated by dividing the number of matches 
    over the length of the true sequence.
    
    Args:
        y (str): true sequence
        p (str): predicted sequence
        alignment_function (object): alignment function from parasail
        matrix (object): matrix object from `parasail.matrix_create`
        open_penalty (int): penalty for opening a gap
        extend_penalty (int): penalty for extending a gap
        
    Returns:
        (float): with the calculated accuracy
    """
    
    if len(p) == 0:
        if len(y) == 0:
            return 1
        else:
            return 0
    
    alignment = alignment_function(p, y, open_penalty, extend_penalty, matrix)
    decoded_cigar = alignment.cigar.decode.decode()
    long_cigar, cigar_counts, cigar_strs = elongate_cigar(decoded_cigar)
    if len(long_cigar) == 0:
        return 0
    
    matches = 0
    for s, i in zip(cigar_strs, cigar_counts):
        if s == '=':
            matches += int(i)
    
    return matches/len(y)