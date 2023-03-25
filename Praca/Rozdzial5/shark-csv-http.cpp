// [...]

static const std::train_data_url = 
"https://raw.githubuserscontent.com/pandas-dev/pandas/master/pandas/
tests/data/iris.csv";
const std::string data_path{"iris.csv"};

// sprawdź istnienie pliku za pomocą std::filesystem
if (!fs::exists(data_path))
{
    if (!utils::DownloadFile(train_data_url, data_path))
    {
        std::cerr << "Unable to download the file " << 
            train_data_url << std::endl;
        return 1;
    }
}

// odczytaj dane z pliku do ciągu znakowego
std::ifstream data_file(data_path);
std::string train_data_str((std::istreambuf_iterator<char>(data_file)), 
                           std::istreambuf_iterator<char>());

// usuń wiersz zawierający etykiety
train_data_str.erase(0, train_data_str.find_first_of("\n") + 1);

// zamień nazwy klas na identyfikatory
train_data_str = 
    std::regex_replace(train_data_str, std::regex("Iris-setosa"), "0");
train_data_str = 
    std::regex_replace(train_data_str, std::regex("Iris-versicolor"), "1");
train_data_str = 
    std::regex_replace(train_data_str, std::regex("Iris-virginica"), "2");

// odczytanie i parsowanie danych
shark::ClassificationDataset train_data;
shark::csvStringToData(train_data, train_data_str, shark::LAST_COLUMN);