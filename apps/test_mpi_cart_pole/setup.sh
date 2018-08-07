make -C ../makefiles/ clean

make -C ../makefiles/ app=test_mpi_cart_pole -j config=prod
