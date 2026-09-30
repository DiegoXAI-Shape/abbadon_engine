# Versión mínima de CMake para hacer la build del programa
cmake_minimum_required(VERSION 3.18)

# Nombre del proyecto, en esté caso abbadon_engine y LANGUAGES CXX es decirle: "Usa C++ como lenguaje"
project(abbadon_engine LANGUAGES CXX)

# Encuentra los paquetes de TorchLib y el argumento REQUIRED es solamente para que CMake sepa que es OBLIGATORIO tenerlo
# Para continuar con la configuración 
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Crea el programa engine a partir del código del archivo src/engine.cpp
add_executable(engine src/engine.cpp src/elements/processors.cpp src/elements/check.cpp)

# Enlaza la librería con el programa 
target_link_libraries(engine ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(engine PRIVATE include)

# Mencionar que versión de C++ usarás
set_property(TARGET engine PROPERTY CXX_STANDARD 20)

# ---------------- Tests ----------------
# Programa aparte (engine_tests) con los tests automáticos. Se corre desde la raíz:
#   ./build/engine_tests    o    ctest --test-dir build --output-on-failure
enable_testing()
add_executable(engine_tests src/tests/test_engine.cpp src/elements/processors.cpp)
target_link_libraries(engine_tests ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(engine_tests PRIVATE include)
set_property(TARGET engine_tests PROPERTY CXX_STANDARD 20)
add_test(NAME engine_tests COMMAND engine_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
