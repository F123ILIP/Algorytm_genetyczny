import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from scipy.spatial.distance import cdist

num_cities = 30
num_cities = int( input( "How many cities do you want: " ) )
population_size = 200
population = int( input( "How large population do you want: " ) )
area_size = 300
crossover_prob = 100
crossover_prob = int( input( "How large probability to crossover: " ) )
mutation_prob_percentage = 5
mutation_prob = int( input( "How large probability of mutation do you want: " ) )


def create_plot( num_cities, area_size ):
    x_min, x_max = -area_size / 2, area_size / 2
    y_min, y_max = -area_size / 2, area_size / 2

    x_coords = np.random.uniform( x_min, x_max, num_cities )
    y_coords = np.random.uniform( y_min, y_max, num_cities )

    plt.figure( figsize = ( 8, 8 ) )
    plt.plot( x_coords, y_coords, 'bo' )
    plt.title( 'Losowe punkty na obszarze' )
    plt.xlabel( 'Współrzędna x' )
    plt.ylabel( 'Współrzędna y' )
    plt.axis( 'equal' )
    plt.grid( True )
    plt.show()

    cities = pd.DataFrame( { 'X': x_coords, 'Y': y_coords } )
    return cities


def create_initial_population( num_cities, population_size ):
    population = []
    for _ in range( population_size ):
        population.append( np.random.permutation( num_cities ) )
    population = pd.DataFrame( population )
    return population

def quality_function( df, dist_tab ):
    total_distances = []
    for i in range( df.shape[0] ):
        route = df.iloc[i].values
        total_distance = sum( dist_tab.iloc[ route[j], route[j+1] ] for j in range( len( route ) - 1 ) )
        total_distance += dist_tab.iloc[ route[-1], route[0] ]
        total_distances.append( total_distance )
    df[ 'total' ] = total_distances
    return df

def calculate_distances( df ):
    distances = cdist( df.values, df.values, 'euclidean' )
    return pd.DataFrame( distances, index = df.index, columns = df.index )

def qf_min( df ):
    df[ 'rank' ] = ( df[ 'total' ].rank( method = 'first' ) - 1 ).astype( int )
    return df

def plot_route( cities, route ):
    plt.figure( figsize = ( 8, 8 ) )
    plt.plot( cities['X'], cities['Y'], 'bo' )
    for i in range( len( route ) - 1 ):
        city1 = route[i]
        city2 = route[i+1]
        if city1 in cities.index and city2 in cities.index:
            x_values = [ cities.loc[ city1, 'X' ], cities.loc[ city2, 'X' ] ]
            y_values = [ cities.loc[ city1, 'Y' ], cities.loc[ city2, 'Y' ] ]
            plt.plot( x_values, y_values, 'r-' )
    x_values = [ cities.loc[ route[-1], 'X' ], cities.loc[ route[0], 'X' ] ]
    y_values = [ cities.loc[ route[-1], 'Y' ], cities.loc[ route[0], 'Y' ] ]

    plt.plot(x_values, y_values, 'r-')
    plt.title( 'Trasa' )
    plt.xlabel( 'Współrzędna x' )
    plt.ylabel( 'Współrzędna y' )
    plt.axis( 'equal' )
    plt.grid( True )
    plt.show()

def plot_best_route( cities, population ):
    best_individual = population.sort_values( by = 'rank' ).iloc[0].values
    plot_route( cities, best_individual )

def cut_bad( df ):
    df = df.sort_values( by = 'rank' ).iloc[0:30]
    return df

def order_crossover( parent1, parent2 ):
    locus_start = random.randint( 0, len( parent1 ) - 1 )
    locus_end = random.randint( locus_start + 1, len( parent1 ) )
    segment = parent1[ locus_start:locus_end ]

    child = []
    for gene in parent2:
        if gene in segment:
            continue
        child.append( gene )

    child[ locus_start:locus_start ] = segment

    return child

def mutate( chromosome ):
    mut1 = random.randint( 0, len( chromosome ) - 2 )
    mut2 = random.randint( mut1 + 1, len( chromosome ) - 1 )
    temp = chromosome.copy()
    chromosome[ mut1 ] = temp[ mut2 ]
    chromosome[ mut2 ] = temp[ mut1 ]
    return chromosome

def succession( population, num_elites, mutation_prob_percentage, crossover_prob ):
    elites = population.sort_values( by = 'rank' ).iloc[ :num_elites, :-2 ]
    children = elites.copy()
    mutation_prob = mutation_prob_percentage / 100.0
    for _ in range( len( elites ) ):
        parent1_idx, parent2_idx = np.random.choice( len( elites ), size = 2, replace = False )
        parent1 = elites.iloc[ parent1_idx ].tolist()
        parent2 = elites.iloc[ parent2_idx ].tolist()

        if random.random() < crossover_prob:
            child = pd.Series( order_crossover( parent1, parent2 ) )
        else:
            child = pd.Series( parent1 )

        if random.random() < mutation_prob:
            child = mutate( child )
        children = pd.concat( [ children, child.to_frame().T ], ignore_index = True )

    return children



pop = create_initial_population( num_cities, population_size )
print( 'pop', pop )
cit = create_plot( num_cities, area_size )
print( 'cit', cit )
di = calculate_distances( cit )
print( 'di', di )
qf = quality_function( pop, di )
print( 'qf', qf )
qfmin = qf_min( qf )
print( 'qfmin', qfmin )
#print( order_crossover( qfmin.iloc[ 0,:-2 ], qfmin.iloc[ 1,:-2 ] ) )
#print( len( order_crossover( qfmin.iloc[ 0,:-2 ], qfmin.iloc[ 1,:-2 ] ) ) )
plot_best_route( cit, qf )

print( cut_bad( qfmin ) )
next_generation = succession( qfmin, num_cities, mutation_prob_percentage, crossover_prob )
qf2 = quality_function( next_generation, di )
qf2min = qf_min( qf2 )
print( cut_bad( qf2 ) )
plot_best_route( cit, qf2 )

tolerance = 0.1
tol = num_cities
condition = abs(qf2.loc[qf2['rank'] == 0, 'total'].values[0] - qf2.loc[qf2['rank'] == tol, 'total'].values[0]) < tolerance

quality_values = []

while( qf2.loc[qf2['rank'] == 0, 'total'].values[0] > 0.4 * qf.loc[qf['rank'] == 0, 'total'].values[0] and
       qf2.loc[qf2['rank'] == 0, 'total'].values[0] != qf2.loc[qf2['rank'] == tol, 'total'].values[0] ):

    next_generation = succession( qf2min, num_cities, mutation_prob_percentage, crossover_prob )
    qf2 = quality_function( next_generation, di )
    qf2min = qf_min( qf2 )
    print( cut_bad( qf2 ) )
    time.sleep( 0.3 )

    best_quality = qf2.loc[ qf2[ 'rank' ] == 0, 'total' ].values[0]
    quality_values.append( best_quality )

plt.figure( figsize = ( 10, 6 ) )
plt.plot( range( len( quality_values ) ), quality_values, marker = 'o', linestyle='-' )
plt.title( 'Malejąca funkcja jakości dla najlepszego osobnika w kolejnych iteracjach' )
plt.xlabel( 'Numer iteracji' )
plt.ylabel( 'Wartość funkcji jakości' )
plt.grid( True )
plt.show()

plot_best_route( cit, qf2 )