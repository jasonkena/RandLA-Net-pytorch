cd util/nearest_neighbors
python setup.py install --home="."
cd ../../

cd util/cpp_wrappers
sh compile_wrappers.sh
cd ../../../
