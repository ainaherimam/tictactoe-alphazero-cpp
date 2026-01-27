#ifndef METRICS_LOGGER_H
#define METRICS_LOGGER_H
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <map>
#include <vector>
#include <iostream>
#include <set>
#include <algorithm>

class MetricsLogger {
private:
    std::string log_dir;
    std::string metrics_path;
    std::string config_path;
    std::ofstream config_file;
    std::set<std::string> all_columns;  // Track ALL columns ever seen
    std::map<std::string, double> current_metrics;
    int write_count;

public:
    MetricsLogger(const std::string& output_dir) : log_dir(output_dir), write_count(0) {
        std::cout << "\n[LOGGER INIT] Creating MetricsLogger for directory: " << output_dir << std::endl;
        
        // Create directory if it doesn't exist
        system(("mkdir -p " + log_dir).c_str());
        
        // Set up file paths
        metrics_path = log_dir + "/metrics.csv";
        config_path = log_dir + "/config.json";
        
        std::cout << "📊 Metrics logging to: " << log_dir << "\n" << std::endl;
        
        // Open config JSON file
        config_file.open(config_path);
    }

    ~MetricsLogger() {
        std::cout << "\n[LOGGER DESTROY] Total writes: " << write_count << std::endl;
        if (config_file.is_open()) {
            config_file.close();
        }
    }
    
    void log_config(const std::map<std::string, std::string>& config) {
        config_file << "{\n";
        bool first = true;
        for (const auto& [key, value] : config) {
            if (!first) config_file << ",\n";
            config_file << "  \"" << key << "\": " << value;
            first = false;
        }
        config_file << "\n}\n";
        config_file.flush();
    }
    
    void log_metrics(const std::map<std::string, double>& metrics) {
        write_count++;
        
        // Add new columns to our tracking set
        bool has_new_columns = false;
        for (const auto& [key, value] : metrics) {
            if (all_columns.find(key) == all_columns.end()) {
                has_new_columns = true;
            }
            all_columns.insert(key);
        }
        
        // Read all existing data if we have new columns
        std::vector<std::map<std::string, std::string>> existing_rows;
        
        if (has_new_columns) {
            std::cout << "[LOGGER] New columns detected, rebuilding CSV..." << std::endl;
            
            std::ifstream infile(metrics_path);
            if (infile.is_open()) {
                std::string line;
                std::vector<std::string> old_columns;
                
                // Read old header
                if (std::getline(infile, line)) {
                    std::stringstream ss(line);
                    std::string col;
                    while (std::getline(ss, col, ',')) {
                        old_columns.push_back(col);
                    }
                }
                
                // Read all data rows
                while (std::getline(infile, line)) {
                    std::stringstream ss(line);
                    std::string value;
                    std::map<std::string, std::string> row;
                    size_t col_idx = 0;
                    
                    while (std::getline(ss, value, ',') && col_idx < old_columns.size()) {
                        if (!value.empty()) {
                            row[old_columns[col_idx]] = value;
                        }
                        col_idx++;
                    }
                    existing_rows.push_back(row);
                }
                infile.close();
            }
        }
        
        // Create sorted column list for consistent order
        std::vector<std::string> column_list(all_columns.begin(), all_columns.end());
        std::sort(column_list.begin(), column_list.end());
        
        // Open file for writing
        std::ofstream outfile;
        
        if (has_new_columns) {
            // Rewrite entire file with new columns
            outfile.open(metrics_path, std::ios::out | std::ios::trunc);
            
            // Write header
            for (size_t i = 0; i < column_list.size(); ++i) {
                if (i > 0) outfile << ",";
                outfile << column_list[i];
            }
            outfile << "\n";
            
            // Write existing rows with new columns
            for (const auto& row : existing_rows) {
                for (size_t i = 0; i < column_list.size(); ++i) {
                    if (i > 0) outfile << ",";
                    auto it = row.find(column_list[i]);
                    if (it != row.end()) {
                        outfile << it->second;
                    }
                }
                outfile << "\n";
            }
            
            std::cout << "[LOGGER] Rebuilt CSV with " << column_list.size() << " columns" << std::endl;
        } else {
            // Just append new row
            outfile.open(metrics_path, std::ios::app);
        }
        
        // Write new metrics row
        for (size_t i = 0; i < column_list.size(); ++i) {
            if (i > 0) outfile << ",";
            auto it = metrics.find(column_list[i]);
            if (it != metrics.end()) {
                outfile << std::fixed << std::setprecision(6) << it->second;
            }
        }
        outfile << "\n";
        
        outfile.flush();
        outfile.close();
        
        // std::cout << "[LOGGER] Write #" << write_count << " complete" << std::endl;
    }
    
    void add_scalar(const std::string& name, double value) {
        current_metrics[name] = value;
    }
    
    void flush_metrics() {
        if (!current_metrics.empty()) {
            log_metrics(current_metrics);
            current_metrics.clear();
        }
    }
};

#endif // METRICS_LOGGER_H