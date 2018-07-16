make -C ../makefiles/ clean

make -C ../makefiles/ app=../apps/test_mpi_cart_pole/cart-pole -j config=prod
