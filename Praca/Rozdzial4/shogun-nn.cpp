using namespace shogun;
// [...]
// preparing input data
std::size_t n = 10000;
SGMatrix<float64_t> x_values(1, static_cast<index_t>(n));
SGVector<float64_t> y_values(static_cast<index_t>(n));
// [...]
auto x = some<CDenseFeatures<float64_t>>(x_values);
auto y = some<CRegressionLabels>(y_values);

// constructing the network architecture
auto dimensions = x->get_num_features();
auto layers = some<CNeuralLayers>();
layers = wrap(layers->input(dimentions));
layers = wrap(layers->rectified_linear(32));
layers = wrap(layers->rectified_linear(16));
layers = wrap(layers->rectified_linear(8));
layers = wrap(layers->linear(1));
auto all_layers = layers->done();

// creating the network
auto network = some<CNeuralNetwork>(all_layers);
network->quick_connect();
network->initialize_neural_network();

// network configuration
network->set_optimization_method(NNOM_GRADIENT_DESCENT);
network->set_gd_mini_batch_size(64);
network->set_l2_coefficient(0.0001);
network->set_max_num_epochs(500);
network->set_epsilon(0.0); // kryterium zbieżności
network->set_gd_learning_rate(0.01);
network->set_gd_momentum(0.5);

// configuring more precise logs from the training
// process
shogun::sg_io->set_log_level(shogun::MSG_DEBUG);

// training
network->set_labels(y);
network->train(x);