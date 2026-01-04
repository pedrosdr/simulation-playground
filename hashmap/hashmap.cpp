#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

int main()
{
    std::unordered_map<int, std::vector<std::string>> map = {
        {0, {"Pedro", "Maria"}},
        {1, {"Jonathan", "Johna"}},
        {2, {"Cabron"}}
    };

    for(std::string s : map[2]) {
        std::cout << s << ", ";
    }
    return 0;
}