
# coding: utf-8

# In[1]:


import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sage.rings.polynomial.toy_variety import coefficient_matrix
import itertools
from brial import *
from IPython.display import clear_output
import random
import copy
from collections import Counter


# In[2]:


def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def pascal_summation(n, a):
    value = 0
    for i in range(a+1):
        add = binomial(n, i)
        value = value + add
    return value

def return_uniqueterms_in_equations(equation_list):
    monomials_appearing = []
    for equation in equation_list:
        for term in equation.terms():
            if term not in monomials_appearing:
                monomials_appearing.append(term)
    monomials_appearing = badsort(monomials_appearing)
    return monomials_appearing

def badsort(mylist):
    for iterations in range(len(mylist)):            
        for i in range(len(mylist)-1):
            if mylist[i] > mylist[i+1]:
                mylist[i], mylist[i+1] = mylist[i+1], mylist[i]
    return mylist
    
def maximal_polynomial():
    term_list = []
    term = 1
    for i in range(len(B.gens())):
        term = term*B.gens()[i]
    term_list.append(term)
    return term_list

def second_best():
    second_best_list = []
    rji = list(B.gens())
    abc = maximal_polynomial()
    for r in rji:
        t = abc[0].set().vars()//r.lm()
        second_best_list.append(t)
    return second_best_list


# In[5]:


def good_equation_generator(highest_degree, highest_terms, equations):
    v = BooleanPolynomialVector()
    l = [B.random_element(degree = highest_degree, terms = highest_terms) for _ in range(equations)]
    _ = [v.append(e) for e in l]
    output = []
    seen = set()
    for value in l:
    # If value has not been encountered yet,
    # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

def term_limit_finder(degree, ring):
    var_count = len(ring.gens())
    term_limit = 0
    for i in range(degree+1):
        add = binomial(var_count, i)
        term_limit = term_limit + add
    return term_limit

def make_system_of_equations_consistent(equation_list): 
    MS = MatrixSpace(GF(2), len(equation_list), 1)
    matrix_answers = MS()
    x_values = []
    ans = []
    for i in range(NUMBER_OF_VARIABLES):
        value = GF(2).random_element()
        x_values.append(value)
    for f in equation_list:
        answer = f(*x_values)
        ans.append(answer)
    for i in range(len(equation_list)):
        matrix_answers[i,0] = ans[i]
        equation_list[i] = equation_list[i] + ans[i]
    return x_values, equation_list 

def generate_monomials(degree_max, ring):
    terms_limit = term_limit_finder(degree_max, ring)
    monomial_list = []
    v_1 = BooleanPolynomialVector()
    l = [ring.random_element(degree = degree_max, terms = terms_limit) for _ in range(1)]
    _ = [v_1.append(e) for e in l]
    f = l[0]
    for i in f:
        monomial_list.append(i)
    return monomial_list

def generate_monomials_high(degree, ring):
    monomials_to_degree = generate_monomials(degree, ring)
    unwanted = generate_monomials(degree-1, ring)
    '''Probably can make this step faster'''
    monomials_of_degree = [e for e in monomials_to_degree if e not in unwanted]
    return monomials_of_degree

def random_monomial_selector(monomial_list):
    output_monomial = monomial_list[ZZ.random_element(0, len(monomial_list))]
    return output_monomial

def detailed_equation_generator(): 
    ''' adjust the goddamn parameters yourself!'''
    num2degvarseqns = int(NUMBER_OF_VARIABLES/2)
    num2degvars = int(NUMBER_OF_VARIABLES/4)
    num1degvarseqns = 2*NUMBER_OF_VARIABLES
    num1degvars = int(NUMBER_OF_VARIABLES/3)
    generated_equation_list = [B.zero()]* EQUATIONS
    '''init'''
    monomial_list_2 = generate_monomials_high(2, B)
    monomial_list_1 = generate_monomials_high(1, B)
    for i in range(num2degvarseqns): # number of 2degree vars equations with 2 degree vars
        for j in range(num2degvars): # number of 2degree vars in each equation
            monomial_to_add = random_monomial_selector(monomial_list_2)
            generated_equation_list[i] = generated_equation_list[i] + monomial_to_add

    for i in range(num1degvarseqns):
        for j in range(num1degvars):
            monomial_to_add = random_monomial_selector(monomial_list_1)
            generated_equation_list[i] = generated_equation_list[i] + monomial_to_add

    # for i in generated_equation_list:
    #     print i
    #     time.sleep(0.1)
    #     clear_output(wait = True)

    monomial_list = generate_monomials(DEGREE_GAP, B)
    xvals, consistent_equation_list = make_system_of_equations_consistent(generated_equation_list)
    
    return consistent_equation_list, monomial_list

def simple_equation_generator(eqn_degree, terms, equations):
    generated_equation_list = good_equation_generator(eqn_degree, terms, equations)
#     monomial_list = generate_monomials(MONOMIAL_DEGREE, B)
    xvals, consistent_equation_list = make_system_of_equations_consistent(generated_equation_list)
    return consistent_equation_list


# In[4]:


def matrix_generator_density(equation_list_input,monomial_list_input, DENSITY):
    length_of_equations = len(equation_list_input)
    length_of_monomials = len(monomial_list_input)
    probability_mat = random_matrix(GF(2), length_of_equations, length_of_monomials, density = DENSITY)
    return probability_mat

def XL_random_prob(matrix, equation_list_input, monomial_list_input):
    new_equation_list = []
    columns = len(monomial_list_input)
    rows = len(equation_list_input)
    for i in range(rows):
        for j in range(columns):
            if matrix[i][j] == 1:
                monomial_to_mutiply = monomial_list_input[j]
                equation_to_mutiply = equation_list_input[i]
                new_equation = monomial_to_mutiply*equation_to_mutiply
                new_equation_list.append(new_equation)
    return new_equation_list

def full_XL(equation_list, monomial_list):
    matrix =  matrix_generator_density(equation_list,monomial_list, 1)
    new_equation_list = XL_random_prob(matrix, equation_list, monomial_list)
    return new_equation_list

def partial_XL(equation_list, monomial_list, density):
    matrix =  matrix_generator_density(equation_list,monomial_list, density)
    new_equation_list = XL_random_prob(matrix, equation_list, monomial_list)
    return new_equation_list

def XL_reject(equation_list_input, monomial_list_input, reject):
    '''To make it work, increase equations or decrease terms'''
    matrix = matrix_generator_density(equation_list_input, monomial_list_input, 0.5)
    new_equation_list = []
    columns = len(monomial_list_input)
    rows = len(equation_list_input)
    for i in range(rows):
        for j in range(columns):
            if matrix[i][j] == 1:
                monomial_to_mutiply = monomial_list_input[j]
                equation_to_mutiply = equation_list_input[i]
                new_equation = monomial_to_mutiply*equation_to_mutiply
                if new_equation.degree() <= reject:
                    new_equation_list.append(new_equation)
    return new_equation_list

def XL_reject_full(equation_list_input, monomial_list_input, reject):
    '''To make it work, increase equations or decrease terms'''
    matrix = matrix_generator_density(equation_list_input, monomial_list_input, 1)
    new_equation_list = []
    columns = len(monomial_list_input)
    rows = len(equation_list_input)
    for i in range(rows):
        for j in range(columns):
            if matrix[i][j] == 1:
                monomial_to_mutiply = monomial_list_input[j]
                equation_to_mutiply = equation_list_input[i]
                new_equation = monomial_to_mutiply*equation_to_mutiply
                if new_equation.degree() <= reject:
                    new_equation_list.append(new_equation)
    return new_equation_list

def coefficient_matrix_generator(equation_list):
    A, v = Sequence(equation_list).coefficient_matrix()
    return A,v

def echelonize_the_matrix(the_matrix):
    echelonized_matrix = the_matrix.__copy__(); echelonized_matrix.echelonize(k = 10)
    return echelonized_matrix

def solver(equation_list):
    complex_matrix, terms = coefficient_matrix_generator(equation_list)
    print 'Rank of matrix: ' + str(complex_matrix.rank())
    time_start = time.time()
    simple_matrix = echelonize_the_matrix(complex_matrix)
    time_end = time.time()
    simple_equations = simple_matrix*terms
    dict_one, dict_zero = find_simple_solutions(simple_equations)
    print 'Solutions found: ' + str(len(dict_one) + len(dict_zero))
    return dict_one, dict_zero, time_end - time_start

def find_simple_solutions(solutions):
    u = []; v = []
    for i in solutions:
        if i[0].has_constant_part() and len(i[0]) == 2 and i[0].degree() <= 2:
            a = i[0]-1
            if a not in u:
                u.append(a)
        if len(i[0]) == 1 and i[0].degree() <= 2:
            b = i[0]
            if b not in v:
                v.append(b)
    d = {key: 1 for key in u}; e = {key: 0 for key in v}
    return d,e


# In[7]:


'''Random fomulas'''
def standard_guassian_elimination():
    print time.time()
    del_cols = []
    del_rows = []
    for y in range(complex_matrix.ncols()):
        counts = list(complex_matrix.column(y)).count(B.one())
        if counts == 1:
            location = list(complex_matrix.column(y)).index(B.one())
            del_cols.append(y)
            del_rows.append(location)

    print del_rows, del_cols
    reduce_matrix = complex_matrix.__copy__()
    reduce_matrix = reduce_matrix.delete_columns(del_cols)
    reduce_matrix = reduce_matrix.delete_rows(del_rows)
    terms = terms.delete_rows(del_cols)
    print reduce_matrix.nrows(), reduce_matrix.ncols()
    simple_matrix = echelonize_the_matrix(reduce_matrix)
    # print simple_matrix*terms
    print time.time()
    
def XL_smart(equation_list, monomial_list, Density):
    matrix = matrix_generator_density(equation_list, monomial_list, Density)
    second_new_equation_list = []
    columns = len(monomial_list)
    rows = len(equation_list)
    for i in range(rows):
        for j in range(columns):
            if matrix[i][j] == 1:
                monomial_to_mutiply = monomial_list[j]
                equation_to_mutiply = equation_list[i]
                new_equation = monomial_to_mutiply*equation_to_mutiply
                second_new_equation_list.append(new_equation)
    return second_new_equation_list

def remove_least_common_monomials():
    removal = 0
    monomials_to_remove = []
    for monomial, counts in  reversed(monomial_counter(new_equation_list).most_common()):
        if removal <= difference - 3:
            removal = removal + counts
            monomials_to_remove.append(monomial)

    new_equation_list_2 = copy.copy(new_equation_list)

    # fastest
    for equation in new_equation_list:
        if len(set(equation.monomials()).intersection(set(monomials_to_remove))) != 0:
            new_equation_list_2.remove(equation)   

    monomial_counts = monomial_counter(new_equation_list_2);print len(monomial_counts), len(new_equation_list_2)
    dict_one, dict_zero = get_ipython().magic(u'time solver(new_equation_list_2)')
    
def monomial_counter(equation_list):
    y = []
    for equation in equation_list:
        for monomial in equation:
            y.append(monomial)
    monomial_counts = Counter(y)
    return monomial_counts

def testing_values():
    testing_value = 100
    monomial_counts = monomial_counter(new_equation_list)
    least_common_items = monomial_counts.most_common()[-testing_value:]
    dict_to_remove = dict((x,y) for x,y in least_common_items)
    items_to_remove = []
    for item in dict_to_remove:
        items_to_remove.append(item)

    print items_to_remove
    print sum(dict_to_remove.values())
    print monomial_counts
    
    proportion_counter = (float(number_equations - sum(dict_to_remove.values())))/(number_monomials - testing_value)
    print proportion_counter
    
def remove_useless_values(equation_list):
    monomial_counts = monomial_counter(equation_list)
    Y = monomial_counts.most_common()
    items_to_remove = []
    for x,y in Y:
        if y == 1:
            items_to_remove.append(x)
    second_new_equation_list = []
    for equation in equation_list:
        append_equation = True
        for monomial in equation.monomials():
            if monomial in items_to_remove:
                append_equation = False
                continue
        print equation, len(second_new_equation_list)
        clear_output(wait = True)
        if append_equation == True:
            second_new_equation_list.append(equation)
    return second_new_equation_list


# In[19]:


NUMBER_OF_VARIABLES = 12
B = BooleanPolynomialRing(NUMBER_OF_VARIABLES,'x', order = 'degrevlex')
MONOMIAL_DEGREE = 1
EQUATION_DEGREE = 2
EQUATIONS = int(2*NUMBER_OF_VARIABLES)
TERMS = 4

consistent_equation_list = simple_equation_generator(EQUATION_DEGREE, TERMS, EQUATIONS)
monomial_list_1degree = generate_monomials(1, B)
monomial_list_2degree = generate_monomials(2, B)


# In[20]:


new_equation_list = XL_reject_full(consistent_equation_list, monomial_list_1degree, 2)
# new_equation_list = partial_XL(consistent_equation_list, monomial_list_1degree, 1)
new_equation_list = partial_XL(new_equation_list, monomial_list_1degree, 0.95)
print len(new_equation_list)
monomial_counts = monomial_counter(new_equation_list); print len(monomial_counts), len(new_equation_list)
difference = len(new_equation_list) - len(monomial_counts); difference


# In[21]:


get_ipython().magic(u'time dict_one, dict_zero, time_taken = solver(new_equation_list)')
print time_taken, len(dict_one)


# In[22]:


get_ipython().magic(u'time ideal(consistent_equation_list).groebner_basis()')

