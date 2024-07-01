#include <iostream>

extern "C"{
void mulMat(const double* X_mat, int X_n, int X_m,
            const double* Y_mat, int Y_n, int Y_m,
            double* answer)
{
	try
	{	
		if (X_m != Y_n){throw std::string{"The number of columns in X must be equal to the number of rows in Y"};}
		
	}
	catch (const std::string& error_message) {
		std::cout << error_message << std::endl;
        }
	for (int i = 0; i < X_n; ++i)
		{
			double * c = answer + i * Y_m;
			for (int j = 0; j < Y_m; ++j)
				c[j] = 0;
			for (int k = 0; k < X_m; ++k)
			{
				const double * b = Y_mat + k * Y_m;
				float a = X_mat[i*X_m + k];
				for (int j = 0; j < Y_m; ++j)
					c[j] += a * b[j];
				
			}
		}
		std::cout << "Resulting matrix:" << std::endl;
   		for (int i = 0; i < X_n; ++i) {
        for (int j = 0; j < Y_m; ++j) {
            std::cout << answer[i * Y_m + j] << " ";
        }
        std::cout << std::endl;
    }
}
}

