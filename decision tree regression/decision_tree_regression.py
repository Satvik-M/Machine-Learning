{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Copy of decision_tree_regression.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "r3cas2_1T98w"
      },
      "source": [
        "# Decision Tree Regression"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "IODliia6U1xO"
      },
      "source": [
        "## Importing the libraries"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "y98nA5UdU6Hf"
      },
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jpjZ43YlU8eI"
      },
      "source": [
        "## Importing the dataset"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pLVaXoYVU_Uy"
      },
      "source": [
        "dataset = pd.read_csv('Position_Salaries.csv')\n",
        "X = dataset.iloc[:, 1:-1].values\n",
        "y = dataset.iloc[:, -1].values"
      ],
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "g16qFkFQVC35"
      },
      "source": [
        "## Training the Decision Tree Regression model on the whole dataset"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "51dDcEip2Ub_",
        "outputId": "77ba4a38-6bf6-456b-d0ee-d38a56d99fe0"
      },
      "source": [
        "from sklearn.tree import DecisionTreeRegressor\r\n",
        "regressor=DecisionTreeRegressor()\r\n",
        "regressor.fit(X,y)"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=None,\n",
              "                      max_features=None, max_leaf_nodes=None,\n",
              "                      min_impurity_decrease=0.0, min_impurity_split=None,\n",
              "                      min_samples_leaf=1, min_samples_split=2,\n",
              "                      min_weight_fraction_leaf=0.0, presort='deprecated',\n",
              "                      random_state=None, splitter='best')"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MQRGPTH3VcOn"
      },
      "source": [
        "## Predicting a new result"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RH-vK8n222Hd",
        "outputId": "3a57111f-f127-4c89-8358-52a21dd0e4bd"
      },
      "source": [
        "print(regressor.predict([[6.5]]))"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[150000.]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ph8ExBj0VkIT"
      },
      "source": [
        "## Visualising the Decision Tree Regression results (higher resolution)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 293
        },
        "id": "hRzKsrQP2_dO",
        "outputId": "0c0c21ba-7e52-4af0-d85a-66a44f17310e"
      },
      "source": [
        "X_grid=np.arange(min(X),max(X),0.01)\r\n",
        "X_grid=X_grid.reshape(len(X_grid),1)\r\n",
        "plt.scatter(X,y,color=\"red\")\r\n",
        "plt.plot(X_grid,regressor.predict(X_grid))\r\n",
        "plt.show()"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7f624c6376d8>]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BkGd4AIV3kUH"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}