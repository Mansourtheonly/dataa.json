#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <unordered_set>

using namespace std;
using json = nlohmann::json;

// Define Chromosome and Population types
using Chromosome = vector<int>;
using Population = vector<Chromosome>;

// Global variables
size_t N_CAPTEURS = 0;
size_t N_EMPLACEMENTS = 0;
int TAILLE_POPULATION = 100;
int MAX_GENERATIONS = 100;  // Increased from 50 to allow more evolution
double MUTATION_RATE = 0.05;
double CROSSOVER_RATE = 0.8;
int TOURNAMENT_SIZE = 3;
double SELECTION_PROBABILITY = 0.75;
int ELITE_COUNT = 5;  // Number of elite solutions to preserve in each generation

vector<pair<double, double>> emplacements;
vector<pair<double, double>> points_interet;
vector<vector<double>> matrice_distance;
vector<double> rayons_capteurs;

// Function Declarations
double calculateDistance(const pair<double, double>& p1, const pair<double, double>& p2);
bool isSolutionFeasible(const vector<int>& solution);
double calculateFitness(const vector<int>& solution);
vector<int> createInitialSolution(int method);
Population createInitialPopulation();
pair<vector<int>, vector<int>> crossover(const vector<int>& parent1, const vector<int>& parent2);
void mutate(vector<int>& solution, bool adaptiveMutation);
vector<int> tournamentSelection(const Population& population, const vector<double>& fitnesses);
void calculateDistanceMatrix();
vector<int> greedySolution();
void afficherResultats(const string& nom_fichier, const Chromosome& meilleur, bool faisable, int generation_finale, long long temps);
void sauvegarderMeilleur(const Chromosome& individu, int generation_finale, const string& nom_fichier, bool feasible, double fitness);

// New structure to store solution stats
struct SolutionStats {
    int generation;
    double best_fitness;
    int sensors_used;
    bool feasible;
};

int main() {
    // Set random seed for reproducibility
    srand(static_cast<unsigned int>(time(nullptr)));
    
    // Load data from json file
    ifstream file("data1.json");
    if (!file.is_open()) {
        cerr << "Error: Could not open data1.json" << endl;
        return 1;
    }

    json data;
    try {
        file >> data;
    } catch (const json::parse_error& e) {
        cerr << "Error parsing JSON: " << e.what() << endl;
        return 1;
    }
    file.close();

    // Extract genetic parameters
    if (data.contains("genetic_params")) {
        json geneticParams = data["genetic_params"];
        TAILLE_POPULATION = geneticParams.value("population_size", TAILLE_POPULATION);
        CROSSOVER_RATE = geneticParams.value("prob_crossover", CROSSOVER_RATE);
        MUTATION_RATE = geneticParams.value("prob_mutation", MUTATION_RATE);
        TOURNAMENT_SIZE = geneticParams.value("tournament_size", TOURNAMENT_SIZE);
        MAX_GENERATIONS = geneticParams.value("max_generations", MAX_GENERATIONS);
        SELECTION_PROBABILITY = geneticParams.value("selection_probability", SELECTION_PROBABILITY);
    } else {
        cerr << "Warning: 'genetic_params' not found in JSON, using default values." << endl;
    }

    // Output files setup
    string fichier_csv = "resultats_improved.csv";
    ofstream csv(fichier_csv);
    csv << "Fichier,Nb_POIs,Nb_Capteurs,Faisable,Capteurs_Utilises,Temps_Execution(ms),Generation_Finale,Fitness\n";

    ofstream solLog("solutions_improved.log", ios::trunc);
    solLog.close();  // Just to clear the file

    // Process data for each file
    for (auto& [filename, bmp_data] : data.items()) {
        if (filename == "genetic_params") continue;

        cout << "\n==============================================" << endl;
        cout << "Processing: " << filename << endl;
        cout << "==============================================" << endl;

        // Clear data for each file
        emplacements.clear();
        points_interet.clear();
        rayons_capteurs.clear();
        matrice_distance.clear();

        // Extract locations
        if (bmp_data.contains("locations") && bmp_data["locations"].is_array()) {
            for (auto& loc : bmp_data["locations"]) {
                if (loc.is_object() && loc.contains("x") && loc.contains("y") && loc["x"].is_number() && loc["y"].is_number()) {
                    emplacements.push_back({loc["x"].get<double>(), loc["y"].get<double>()});
                }
            }
            N_EMPLACEMENTS = emplacements.size();
            cout << "Loaded " << N_EMPLACEMENTS << " potential sensor locations" << endl;
        } else {
            cerr << "Warning: 'locations' not found or invalid in " << filename << ", skipping." << endl;
            continue;
        }

        // Extract POIs
        if (bmp_data.contains("pois") && bmp_data["pois"].is_array()) {
            for (auto& poi : bmp_data["pois"]) {
                if (poi.is_object() && poi.contains("x") && poi.contains("y") && poi["x"].is_number() && poi["y"].is_number()) {
                    points_interet.push_back({poi["x"].get<double>(), poi["y"].get<double>()});
                }
            }
            cout << "Loaded " << points_interet.size() << " points of interest" << endl;
        } else {
            cerr << "Warning: 'pois' not found or invalid in " << filename << ", skipping." << endl;
            continue;
        }

        // Extract sensors
        if (bmp_data.contains("sensors") && bmp_data["sensors"].is_array()) {
            for (auto& sensor : bmp_data["sensors"]) {
                if (sensor.is_object() && sensor.contains("range") && sensor["range"].is_number()) {
                    rayons_capteurs.push_back(sensor["range"].get<double>());
                }
            }
            N_CAPTEURS = rayons_capteurs.size();
            cout << "Loaded " << N_CAPTEURS << " sensor types" << endl;
        } else {
            cerr << "Warning: 'sensors' not found or invalid in " << filename << ", skipping." << endl;
            continue;
        }

        // Calculate distance matrix between locations and POIs
        calculateDistanceMatrix();
        
        // Try initial greedy solution
        vector<int> greedy = greedySolution();
        bool greedyFeasible = isSolutionFeasible(greedy);
        int greedySensors = count_if(greedy.begin(), greedy.end(), [](int g) { return g != 0; });
        cout << "Greedy solution: " << (greedyFeasible ? "Feasible" : "Infeasible") 
             << ", uses " << greedySensors << " sensors" << endl;

        // Genetic Algorithm
        auto debut = chrono::high_resolution_clock::now();
        
        // Create initial population with mixture of random and greedy solutions
        Population population = createInitialPopulation();
        
        // Add greedy solution if it's feasible
        if (greedyFeasible) {
            population[0] = greedy; // Replace first solution with greedy
        }
        
        // Track progress
        vector<double> fitnesses(TAILLE_POPULATION);
        vector<SolutionStats> stats;
        
        Chromosome meilleur;
        double meilleur_fitness = -1.0;
        bool faisable = false;
        int generation_finale = 0;
        
        int stagnation_count = 0;
        bool allSolutionsFeasible = false;
        int noImprovementCount = 0;
        
        // Main evolution loop
        for (int generation = 0; generation < MAX_GENERATIONS; generation++) {
            // Calculate fitness for all chromosomes
            for (int i = 0; i < TAILLE_POPULATION; i++) {
                fitnesses[i] = calculateFitness(population[i]);
            }
            
            // Find best solution in current generation
            int best_index = max_element(fitnesses.begin(), fitnesses.end()) - fitnesses.begin();
            double best_gen_fitness = fitnesses[best_index];
            bool best_gen_feasible = isSolutionFeasible(population[best_index]);
            int best_gen_sensors = count_if(population[best_index].begin(), population[best_index].end(), 
                                           [](int g) { return g != 0; });
            
            // Update overall best solution if better found
            if (generation == 0 || best_gen_fitness > meilleur_fitness) {
                meilleur_fitness = best_gen_fitness;
                meilleur = population[best_index];
                faisable = best_gen_feasible;
                generation_finale = generation;
                noImprovementCount = 0;
            } else {
                noImprovementCount++;
            }
            
            // Log statistics
            stats.push_back({generation, best_gen_fitness, best_gen_sensors, best_gen_feasible});
            
            // Progress feedback every 10 generations
            if (generation % 10 == 0) {
                cout << "Generation " << generation 
                     << " - Best fitness: " << best_gen_fitness
                     << " - Feasible: " << (best_gen_feasible ? "Yes" : "No")
                     << " - Sensors: " << best_gen_sensors << endl;
            }
            
            // Check for convergence conditions
            if (noImprovementCount >= 20) {
                cout << "Early stopping due to no improvement in 20 generations" << endl;
                break;
            }
            
            // Create next generation
            Population nouvelle_population;
            
            // Elitism - add best solutions directly to next generation
            vector<pair<double, int>> fitness_indices;
            for (int i = 0; i < TAILLE_POPULATION; i++) {
                fitness_indices.push_back({fitnesses[i], i});
            }
            sort(fitness_indices.begin(), fitness_indices.end(), greater<pair<double, int>>());
            
            for (int i = 0; i < min(ELITE_COUNT, TAILLE_POPULATION); i++) {
                nouvelle_population.push_back(population[fitness_indices[i].second]);
            }
            
            // Generate rest of population through selection, crossover, mutation
            while (nouvelle_population.size() < TAILLE_POPULATION) {
                vector<int> parent1 = tournamentSelection(population, fitnesses);
                vector<int> parent2 = tournamentSelection(population, fitnesses);
                
                if (static_cast<double>(rand()) / RAND_MAX < CROSSOVER_RATE) {
                    pair<vector<int>, vector<int>> enfants = crossover(parent1, parent2);
                    nouvelle_population.push_back(enfants.first);
                    if (nouvelle_population.size() < TAILLE_POPULATION) {
                        nouvelle_population.push_back(enfants.second);
                    }
                } else {
                    nouvelle_population.push_back(parent1);
                    if (nouvelle_population.size() < TAILLE_POPULATION) {
                        nouvelle_population.push_back(parent2);
                    }
                }
            }
            
            // Apply mutation to new population
            for (int i = ELITE_COUNT; i < TAILLE_POPULATION; i++) {  // Skip elite solutions
                mutate(nouvelle_population[i], true);
            }
            
            // Replace old population
            population = nouvelle_population;
            
            // Dynamic adjustment of mutation rate based on population diversity
            double diversity = 0.0;
            for (int i = 0; i < N_CAPTEURS; i++) {
                int active_count = 0;
                for (const auto& solution : population) {
                    if (solution[i] == 1) active_count++;
                }
                double gene_diversity = min(active_count, (int)population.size() - active_count) / (double)population.size();
                diversity += gene_diversity;
            }
            diversity /= N_CAPTEURS;
            
            // Adjust mutation rate if diversity is low
            if (diversity < 0.1) {
                MUTATION_RATE = min(0.3, MUTATION_RATE * 1.5);
            } else {
                MUTATION_RATE = max(0.01, MUTATION_RATE * 0.9);
            }
        }
        
        auto fin = chrono::high_resolution_clock::now();
        auto temps = chrono::duration_cast<chrono::milliseconds>(fin - debut);
        
        int capteurs_utilises = count_if(meilleur.begin(), meilleur.end(), [](int g) { return g != 0; });
        
        // Display and save results
        afficherResultats(filename, meilleur, faisable, generation_finale, temps.count());
        sauvegarderMeilleur(meilleur, generation_finale, filename, faisable, meilleur_fitness);
        
        csv << filename << ","
            << points_interet.size() << ","
            << N_CAPTEURS << ","
            << (faisable ? "Oui" : "Non") << ","
            << capteurs_utilises << ","
            << temps.count() << ","
            << generation_finale << ","
            << meilleur_fitness << "\n";
        
        // Print coverage analysis for the best solution
        if (faisable) {
            cout << "\nBest solution coverage analysis:" << endl;
            int total_covered_twice = 0;
            vector<int> coverage_count(points_interet.size(), 0);
            
            for (size_t j = 0; j < points_interet.size(); j++) {
                for (size_t i = 0; i < meilleur.size(); i++) {
                    if (meilleur[i] == 1) {
                        if (matrice_distance[i][j] <= rayons_capteurs[i]) {
                            coverage_count[j]++;
                        }
                    }
                }
                if (coverage_count[j] > 1) total_covered_twice++;
            }
            cout << "Points covered by multiple sensors: " << total_covered_twice 
                 << " (" << (total_covered_twice * 100.0 / points_interet.size()) << "%)" << endl;
        }
    }
    
    csv.close();
    cout << "\n✅ All experiments processed. Results saved in " << fichier_csv << endl;
    
    return 0;
}

double calculateDistance(const pair<double, double>& p1, const pair<double, double>& p2) {
    return sqrt(pow(p1.first - p2.first, 2) + pow(p1.second - p2.second, 2));
}

void calculateDistanceMatrix() {
    matrice_distance.resize(emplacements.size());
    for (size_t i = 0; i < emplacements.size(); i++) {
        matrice_distance[i].resize(points_interet.size());
        for (size_t j = 0; j < points_interet.size(); j++) {
            matrice_distance[i][j] = calculateDistance(emplacements[i], points_interet[j]);
        }
    }
}

bool isSolutionFeasible(const vector<int>& solution) {
    vector<bool> covered(points_interet.size(), false);
    
    for (size_t j = 0; j < points_interet.size(); j++) {
        for (size_t i = 0; i < solution.size(); i++) {
            if (solution[i] == 1) {
                if (matrice_distance[i][j] <= rayons_capteurs[i]) {
                    covered[j] = true;
                    break;
                }
            }
        }
    }
    
    return count(covered.begin(), covered.end(), false) == 0;
}

double calculateFitness(const vector<int>& solution) {
    // Count covered POIs and calculate coverage ratio
    vector<bool> covered(points_interet.size(), false);
    int covered_count = 0;
    
    for (size_t j = 0; j < points_interet.size(); j++) {
        for (size_t i = 0; i < solution.size(); i++) {
            if (solution[i] == 1) {
                if (matrice_distance[i][j] <= rayons_capteurs[i]) {
                    covered[j] = true;
                    covered_count++;
                    break;
                }
            }
        }
    }
    
    double coverage_ratio = static_cast<double>(covered_count) / points_interet.size();
    
    // Count used sensors
    int sensor_count = count_if(solution.begin(), solution.end(), [](int g) { return g != 0; });
    if (sensor_count == 0) return 0.0; // Avoid division by zero
    
    // Calculate fitness - balance between coverage and sensor count
    if (coverage_ratio >= 0.99) { // Full or almost full coverage
        // Higher score for fewer sensors when coverage is complete
        return 10.0 + (1.0 / sensor_count);
    } else {
        // When coverage is incomplete, prioritize coverage but still 
        // consider sensor count as a secondary objective
        return coverage_ratio - (0.01 * sensor_count / solution.size());
    }
}

vector<int> createInitialSolution(int method) {
    vector<int> solution(N_CAPTEURS, 0);
    
    if (method == 0) {
        // Random solution
        for (size_t i = 0; i < N_CAPTEURS; i++) {
            solution[i] = (rand() % 100 < 50) ? 1 : 0;
        }
    } else if (method == 1) {
        // All sensors active
        fill(solution.begin(), solution.end(), 1);
    } else if (method == 2) {
        // Sparse random solution
        for (size_t i = 0; i < N_CAPTEURS; i++) {
            solution[i] = (rand() % 100 < 20) ? 1 : 0;
        }
    } else if (method == 3) {
        // Pick random subset of sensors
        int num_active = 1 + rand() % (N_CAPTEURS / 2);
        unordered_set<int> active_sensors;
        while (active_sensors.size() < num_active) {
            active_sensors.insert(rand() % N_CAPTEURS);
        }
        for (int idx : active_sensors) {
            solution[idx] = 1;
        }
    }
    
    return solution;
}

vector<int> greedySolution() {
    vector<int> solution(N_CAPTEURS, 0);
    vector<bool> covered(points_interet.size(), false);
    int total_covered = 0;
    
    while (total_covered < points_interet.size()) {
        // Find sensor that covers most uncovered POIs
        int best_sensor = -1;
        int best_coverage = -1;
        
        for (size_t i = 0; i < N_CAPTEURS; i++) {
            if (solution[i] == 0) {  // Only consider unused sensors
                int coverage = 0;
                for (size_t j = 0; j < points_interet.size(); j++) {
                    if (!covered[j] && matrice_distance[i][j] <= rayons_capteurs[i]) {
                        coverage++;
                    }
                }
                
                if (coverage > best_coverage) {
                    best_coverage = coverage;
                    best_sensor = i;
                }
            }
        }
        
        // If we couldn't find a sensor that improves coverage, break out
        if (best_sensor == -1 || best_coverage == 0) break;
        
        // Add best sensor to solution
        solution[best_sensor] = 1;
        
        // Update coverage
        int newly_covered = 0;
        for (size_t j = 0; j < points_interet.size(); j++) {
            if (!covered[j] && matrice_distance[best_sensor][j] <= rayons_capteurs[best_sensor]) {
                covered[j] = true;
                newly_covered++;
            }
        }
        
        total_covered += newly_covered;
        
        // Break if we're not making progress
        if (newly_covered == 0) break;
    }
    
    return solution;
}

Population createInitialPopulation() {
    Population population(TAILLE_POPULATION);
    
    // Create diverse initial population with different initialization methods
    for (int i = 0; i < TAILLE_POPULATION; i++) {
        if (i == 0) {
            // First solution is greedy
            population[i] = greedySolution();
        } else if (i < TAILLE_POPULATION / 4) {
            // 25% are random with 50% active sensors
            population[i] = createInitialSolution(0);
        } else if (i < TAILLE_POPULATION / 2) {
            // 25% are all sensors active
            population[i] = createInitialSolution(1);
        } else if (i < 3 * TAILLE_POPULATION / 4) {
            // 25% are sparse random (20% active)
            population[i] = createInitialSolution(2);
        } else {
            // 25% are random subsets
            population[i] = createInitialSolution(3);
        }
    }
    
    return population;
}

pair<vector<int>, vector<int>> crossover(const vector<int>& parent1, const vector<int>& parent2) {
    vector<int> child1(N_CAPTEURS);
    vector<int> child2(N_CAPTEURS);
    
    // Choose crossover method randomly
    int method = rand() % 3;
    
    if (method == 0) {
        // Single-point crossover
        int crossover_point = rand() % N_CAPTEURS;
        for (size_t i = 0; i < N_CAPTEURS; i++) {
            if (i < crossover_point) {
                child1[i] = parent1[i];
                child2[i] = parent2[i];
            } else {
                child1[i] = parent2[i];
                child2[i] = parent1[i];
            }
        }
    } else if (method == 1) {
        // Uniform crossover
        for (size_t i = 0; i < N_CAPTEURS; i++) {
            if (rand() % 2 == 0) {
                child1[i] = parent1[i];
                child2[i] = parent2[i];
            } else {
                child1[i] = parent2[i];
                child2[i] = parent1[i];
            }
        }
    } else {
        // Adaptive crossover - focuses on preserving active sensors
        for (size_t i = 0; i < N_CAPTEURS; i++) {
            if (parent1[i] == 1 && parent2[i] == 1) {
                // Both parents have active sensor - high probability of keeping it
                child1[i] = (rand() % 100 < 90) ? 1 : 0;
                child2[i] = (rand() % 100 < 90) ? 1 : 0;
            } else if (parent1[i] == 0 && parent2[i] == 0) {
                // Both inactive - low probability of activating
                child1[i] = (rand() % 100 < 10) ? 1 : 0;
                child2[i] = (rand() % 100 < 10) ? 1 : 0;
            } else {
                // One active, one inactive - 50/50
                child1[i] = (rand() % 2 == 0) ? parent1[i] : parent2[i];
                child2[i] = (rand() % 2 == 0) ? parent2[i] : parent1[i];
            }
        }
    }
    
    return make_pair(child1, child2);
}

void mutate(vector<int>& solution, bool adaptiveMutation) {
    // Get solution stats
    bool feasible = isSolutionFeasible(solution);
    int active_sensors = count_if(solution.begin(), solution.end(), [](int g) { return g != 0; });
    
    // Use different mutation strategies based on solution quality
    double base_rate = MUTATION_RATE;
    
    if (adaptiveMutation) {
        if (!feasible) {
            // For infeasible solutions, increase mutation rate and bias toward activating sensors
            base_rate *= 2.0;
            for (size_t i = 0; i < N_CAPTEURS; i++) {
                if (static_cast<double>(rand()) / RAND_MAX < base_rate) {
                    if (solution[i] == 0) {
                        // Higher chance to activate sensors for infeasible solutions
                        solution[i] = (rand() % 100 < 70) ? 1 : 0;
                    } else {
                        // Lower chance to deactivate
                        solution[i] = (rand() % 100 < 30) ? 0 : 1;
                    }
                }
            }
        } else {
            // For feasible solutions, try to minimize sensor count
            for (size_t i = 0; i < N_CAPTEURS; i++) {
                if (static_cast<double>(rand()) / RAND_MAX < base_rate) {
                    if (solution[i] == 1) {
                        // Try deactivating a sensor
                        solution[i] = 0;
                        
                        // If solution becomes infeasible, revert
                        if (!isSolutionFeasible(solution)) {
                            solution[i] = 1;
                        }
                    } else {
                        // Small chance to activate an unused sensor
                        solution[i] = (rand() % 100 < 20) ? 1 : 0;
                    }
                }
            }
        }
    } else {
        // Standard mutation
        for (size_t i = 0; i < N_CAPTEURS; i++) {
            if (static_cast<double>(rand()) / RAND_MAX < base_rate) {
                solution[i] = 1 - solution[i];
            }
        }
    }
}

vector<int> tournamentSelection(const Population& population, const vector<double>& fitnesses) {
    // Select TOURNAMENT_SIZE random individuals
    vector<int> tournament_indices;
    for (int i = 0; i < TOURNAMENT_SIZE; i++) {
        tournament_indices.push_back(rand() % population.size());
    }
    
    // Find the best individual in the tournament
    int best_index = tournament_indices[0];
    for (size_t i = 1; i < tournament_indices.size(); i++) {
        if (fitnesses[tournament_indices[i]] > fitnesses[best_index]) {
            best_index = tournament_indices[i];
        }
    }
    
    // Apply selection probability - usually select the best
    if (static_cast<double>(rand()) / RAND_MAX < SELECTION_PROBABILITY) {
        return population[best_index];
    } else {
        // Occasionally select a random one
        int random_index = tournament_indices[rand() % tournament_indices.size()];
        return population[random_index];
    }
}

void afficherResultats(const string& nom_fichier, const Chromosome& meilleur, bool faisable, int generation_finale, long long temps) {
    cout << "\n====== Results for " << nom_fichier << " ======" << endl;
    cout << "Solution found in generation: " << generation_finale << endl;
    cout << "Feasible: " << (faisable ? "Yes" : "No") << endl;
    
    int used_sensors = 0;
    cout << "Sensors used: ";
    for (size_t i = 0; i < meilleur.size(); i++) {
        if (meilleur[i] != 0) {
            cout << i + 1 << " ";
            used_sensors++;
        }
    }
    cout << "(" << used_sensors << " total)" << endl;
    cout << "Execution time: " << temps << " ms" << endl;
}

void sauvegarderMeilleur(const Chromosome& individu, int generation_finale, const string& nom_fichier, bool feasible, double fitness) {
    ofstream log("solutions_improved.log", ios::app);
    log << "====== Results for " << nom_fichier << " ======" << endl;
    log << "Solution found in generation: " << generation_finale << endl;
    log << "Feasible: " << (feasible ? "Yes" : "No") << endl;
    log << "Fitness: " << fitness << endl;
    
    int used_sensors = 0;
    log << "Sensors used: ";
    for (size_t i = 0; i < individu.size(); i++) {
        if (individu[i] != 0) {
            log << i + 1 << " ";
            used_sensors++;
        }
    }
    log << "(" << used_sensors << " total)" << endl;
    
    // Add coverage details
    vector<bool> covered(points_interet.size(), false);
    for (size_t j = 0; j < points_interet.size(); j++) {
        for (size_t i = 0; i < individu.size(); i++) {
            if (individu[i] != 0) {
                if (matrice_distance[i][j] <= rayons_capteurs[i]) {
                    covered[j] = true;
                    break;
                }
            }
        }
    }
    
    int coverage_count = count(covered.begin(), covered.end(), true);
    log << "POIs covered: " << coverage_count << "/" << points_interet.size() 
        << " (" << (coverage_count * 100.0 / points_interet.size()) << "%)" << endl;
    log << "-------------------------------------------" << endl;
}