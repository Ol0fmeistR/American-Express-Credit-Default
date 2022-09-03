### American-Express-Credit-Default
<b>Business problem:</b> Given anonymized transaction data with 190
features for 500000 American Express customers, the objective is
to identify which customer is likely to default in the next 180 days <br>
<br>
<b>Solution:</b> Ensembled a LightGBM 'dart' booster model with a 2-layer deep CNN. Both models involved significant feature engineering with the LightGBM
model optimized for minimizing logloss and the CNN(activation: Mish) model optimized for reducing focal loss. Used weight of evidence encoding to generate new features for the LightGBM model and calculated payment/balance statement related features for the Keras CNN. <br>
<br>
