/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Number of priticles
  num_particles = 100;
  
  // random number generator
  std::default_random_engine gen;
  
  // Normal distributions for initial coordinates
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  // Initialize the weights and particles
  particles.resize(num_particles);
  weights.resize(num_particles);
  for (int i = 0; i < num_particles; i++)
  {
    double cur_x = dist_x(gen);
    double cur_y = dist_y(gen);
    double cur_theta = dist_theta(gen);
    double cur_weight = 1.;
    
    Particle cur_particle {/*id*/ i, /*x*/ cur_x, /*y*/ cur_y, /* theta */ cur_theta, /* weight */ cur_weight};
    particles[i] = cur_particle;
    weights[i] = cur_weight;
  }
  
  // Mark object as initialized
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Random number generator
  std::default_random_engine gen;
  
  // Update position for each particle
  for (Particle& cur_particle : particles)
  {
    // Calculate the mean for the x, y and theta
    double new_x_mean = cur_particle.x + (std::abs(yaw_rate) < 0.0000001 ?
      velocity * delta_t * cos(cur_particle.theta) :
      velocity / yaw_rate * (sin(cur_particle.theta + yaw_rate * delta_t) - sin(cur_particle.theta)));
    double new_y_mean = cur_particle.y + (std::abs(yaw_rate) < 0.0000001 ?
      velocity * delta_t * sin(cur_particle.theta) :
      velocity / yaw_rate * (cos(cur_particle.theta) - cos(cur_particle.theta + yaw_rate * delta_t)));
    double new_theta_mean = cur_particle.theta + yaw_rate * delta_t;
    
    // Normal distributions for new coordinates
    std::normal_distribution<double> dist_x(new_x_mean, std_pos[0]);
    std::normal_distribution<double> dist_y(new_y_mean, std_pos[1]);
    std::normal_distribution<double> dist_theta(new_theta_mean, std_pos[2]);
    
    // Update particle with new estimate
    cur_particle.x = dist_x(gen);
    cur_particle.y = dist_y(gen);
    cur_particle.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // Iterate through observations
  for (LandmarkObs& cur_obs : observations)
  {
    // Distance to nearest prediction
    double min_distance = INFINITY;
    // Nearest prediction id
    int min_id = 0;
    
    // For each predicted landmark
    for (const LandmarkObs& cur_pred : predicted)
    {
      // Distance from current observation to current prediction
      double cur_dist = dist(cur_pred.x, cur_pred.y, cur_obs.x, cur_obs.y);
      
      // If current distance is less then last known minimal distance,
      // update the minimum distance and nearest prediction id
      if (cur_dist < min_distance)
      {
        min_distance = cur_dist;
        min_id = cur_pred.id;
      }
    }
    
    // Assign nearest prediction id to currently known nearest id
    cur_obs.id = min_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  // For every particle
  for (int particle_index = 0; particle_index < particles.size(); particle_index++)
  {
    // Current particle
    Particle& cur_particle = particles[particle_index];
    
    // Convert each observation in map's coordinate system
    std::vector<LandmarkObs> observations_t;
    
    // For each observation
    for (const LandmarkObs& cur_obs : observations)
    {
      // Translate and rotate each landmark from vehicle coordinate system to map's coordinate system
      LandmarkObs landmark_t;
      landmark_t.id = cur_obs.id;
      landmark_t.x = cur_obs.x * cos(cur_particle.theta) - cur_obs.y * sin(cur_particle.theta) + cur_particle.x;
      landmark_t.y = cur_obs.x * sin(cur_particle.theta) + cur_obs.y * cos(cur_particle.theta) + cur_particle.y;
      observations_t.push_back(landmark_t);
    }
    
    // Obtain predicted landmark list
    std::vector<LandmarkObs> predicted;
    
    // For each landmark
    for (const Map::single_landmark_s& landmark : map_landmarks.landmark_list) {
      // If landmark is within sensor range
      if (dist(landmark.x_f, landmark.y_f, cur_particle.x, cur_particle.y) <= sensor_range) {
        // Convert landmark to a predicted observation
        LandmarkObs prediction {landmark.id_i, landmark.x_f, landmark.y_f};
        predicted.push_back(prediction);
      }
    }
    
    // Associate every observation with predicted landmark location
    dataAssociation(predicted, observations_t);
    
    // Calculate the new weight
    double new_weight_product = 1;
    
    // For each observation
    for (const LandmarkObs& cur_obs : observations_t)
    {
      // Assumption: index of landmark in a map is equal to id of landmark - 1
      // Get current prediction
      Map::single_landmark_s cur_pred = map_landmarks.landmark_list[cur_obs.id - 1];
      
      // Differences in x and y between measurement and prediction
      double dx = cur_obs.x - cur_pred.x_f;
      double dy = cur_obs.y - cur_pred.y_f;
      
      // Calculate the new weight
      double new_weight = 1 / (M_PI * 2 * std_landmark[0] * std_landmark[1]) *
        std::exp(-1 * (pow(dx, 2) / pow(std_landmark[0], 2) + pow(dy, 2) / pow(std_landmark[1], 2)));
      
      // Multiply running product of weights by the new weight
      new_weight_product *= new_weight;
    }
    
    // Assign the new weight to the particle
    cur_particle.weight = new_weight_product;
    
    // Assign weight to the list of weights
    weights[particle_index] = new_weight_product;
  }
}

void ParticleFilter::resample() {
  // Create a discrete distribution and a generator
  std::default_random_engine gen;
  std::discrete_distribution<int> distribution {weights.begin(), weights.end()};
  
  // Create vector for new particles
  std::vector<Particle> new_particles;
  
  // Draw new particles based on distribution
  for (int i = 0; i < num_particles; i++) {
    // Generate particle index using the distribution
    int cur_particle_i = distribution(gen);
    
    // Get a particle from the list of particles
    Particle cur_particle = particles[cur_particle_i];
    
    // Push back particle to the list of new particles
    new_particles.push_back(cur_particle);
  }
  
  // Make new particles current particles
  particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
