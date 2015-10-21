from sklearn import preprocessing
import numpy as np
import pandas as pd

from base_feature_transformer import BaseFeatureTransformer
from .. import feature_utilities

NA_VALUE = -1


class CurrentWorkingFeatureTransformer(BaseFeatureTransformer):
    DATE_FEATURES = ['VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159', 'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169', 'VAR_0176', 'VAR_0177', 'VAR_0178', 'VAR_0179', 'VAR_0204', 'VAR_0217']
    FULL_DATE_FEATURES = ['VAR_0073', 'VAR_0075', 'VAR_0217']
    
    TO_CROSS = [('VAR_0644', 'VAR_0645'),
                ('VAR_1434', 'VAR_1435'),
                ('VAR_1617', 'VAR_1618'),
                ('VAR_1433', 'VAR_1434'),
                ('VAR_0581', 'VAR_0582'),
                ('VAR_0594', 'VAR_0595'),
                ('VAR_0664', 'VAR_0665'),
                ('VAR_1851', 'VAR_1852'),
                ('VAR_0662', 'VAR_0663'),
                ('VAR_1050', 'VAR_1051'),
                ('VAR_1096', 'VAR_1097'),
                ('VAR_1094', 'VAR_1095'),
                ('VAR_1669', 'VAR_1670'),
                ('VAR_1343', 'VAR_1344'),
                ('VAR_1677', 'VAR_1678'),
                ('VAR_1694', 'VAR_1695'),
                ('VAR_0057', 'VAR_0058'),
                ('VAR_1682', 'VAR_1683'),
                ('VAR_1698', 'VAR_1699'),
                ('VAR_1681', 'VAR_1682'),
                ('VAR_1697', 'VAR_1698'),
                ('VAR_0696', 'VAR_0697'),
                ('VAR_1676', 'VAR_1677'),
                ('VAR_1693', 'VAR_1694'),
                ('VAR_0979', 'VAR_0980'),
                ('VAR_1744', 'VAR_1745'),
                ('VAR_1713', 'VAR_1714'),
                ('VAR_1407', 'VAR_1408'),
                ('VAR_1675', 'VAR_1676'),
                ('VAR_1692', 'VAR_1693'),
                ('VAR_1065', 'VAR_1066'),
                ('VAR_1586', 'VAR_1587'),
                ('VAR_1069', 'VAR_1070'),
                ('VAR_0994', 'VAR_0995'),
                ('VAR_0046', 'VAR_0047'),
                ('VAR_1430', 'VAR_1431'),
                ('VAR_1621', 'VAR_1622'),
                ('VAR_1431', 'VAR_1432'),
                ('VAR_1680', 'VAR_1681'),
                ('VAR_1594', 'VAR_1595'),
                ('VAR_1595', 'VAR_1596'),
                ('VAR_1696', 'VAR_1697'),
                ('VAR_1619', 'VAR_1620'),
                ('VAR_1620', 'VAR_1621'),
                ('VAR_1447', 'VAR_1448'),
                ('VAR_1446', 'VAR_1447'),
                ('VAR_1596', 'VAR_1597'),
                ('VAR_1196', 'VAR_1197'),
                ('VAR_1406', 'VAR_1407'),
                ('VAR_1898', 'VAR_1899'),
                ('VAR_1645', 'VAR_1646'),
                ('VAR_1066', 'VAR_1067'),
                ('VAR_0050', 'VAR_0051'),
                ('VAR_0484', 'VAR_0485'),
                ('VAR_1722', 'VAR_1723'),
                ('VAR_1591', 'VAR_1592'),
                ('VAR_1641', 'VAR_1642'),
                ('VAR_1823', 'VAR_1824'),
                ('VAR_1467', 'VAR_1468'),
                ('VAR_0047', 'VAR_0048'),
                ('VAR_1587', 'VAR_1588'),
                ('VAR_1070', 'VAR_1071'),
                ('VAR_1061', 'VAR_1062'),
                ('VAR_0546', 'VAR_0547'),
                ('VAR_0932', 'VAR_0933'),
                ('VAR_0962', 'VAR_0963'),
                ('VAR_1752', 'VAR_1753'),
                ('VAR_1688', 'VAR_1689'),
                ('VAR_0602', 'VAR_0603'),
                ('VAR_1743', 'VAR_1744'),
                ('VAR_1599', 'VAR_1600'),
                ('VAR_0548', 'VAR_0549'),
                ('VAR_0497', 'VAR_0498'),
                ('VAR_1437', 'VAR_1438'),
                ('VAR_0641', 'VAR_0642'),
                ('VAR_1498', 'VAR_1499'),
                ('VAR_1249', 'VAR_1250'),
                ('VAR_1244', 'VAR_1245'),
                ('VAR_0601', 'VAR_0602'),
                ('VAR_1466', 'VAR_1467'),
                ('VAR_1687', 'VAR_1688'),
                ('VAR_0378', 'VAR_0379'),
                ('VAR_1490', 'VAR_1491'),
                ('VAR_1468', 'VAR_1469'),
                ('VAR_1287', 'VAR_1288'),
                ('VAR_0485', 'VAR_0486'),
                ('VAR_1615', 'VAR_1616'),
                ('VAR_1613', 'VAR_1614'),
                ('VAR_0068', 'VAR_0069'),
                ('VAR_1646', 'VAR_1647'),
                ('VAR_0591', 'VAR_0592'),
                ('VAR_1771', 'VAR_1772'),
                ('VAR_1672', 'VAR_1673'),
                ('VAR_1062', 'VAR_1063'),
                ('VAR_1487', 'VAR_1488'),
                ('VAR_1787', 'VAR_1788'),
                ('VAR_1341', 'VAR_1342'),
                ('VAR_1799', 'VAR_1800'),
                ('VAR_0193', 'VAR_0194'),
                ('VAR_1756', 'VAR_1757'),
                ('VAR_1637', 'VAR_1638'),
                ('VAR_1441', 'VAR_1442'),
                ('VAR_1812', 'VAR_1813'),
                ('VAR_1479', 'VAR_1480'),
                ('VAR_1804', 'VAR_1805'),
                ('VAR_1283', 'VAR_1284'),
                ('VAR_1592', 'VAR_1593'),
                ('VAR_1642', 'VAR_1643'),
                ('VAR_1603', 'VAR_1604'),
                ('VAR_1432', 'VAR_1433'),
                ('VAR_1813', 'VAR_1814'),
                ('VAR_0387', 'VAR_0388'),
                ('VAR_1268', 'VAR_1269'),
                ('VAR_1783', 'VAR_1784'),
                ('VAR_0056', 'VAR_0057'),
                ('VAR_0694', 'VAR_0695'),
                ('VAR_0067', 'VAR_0068'),
                ('VAR_1485', 'VAR_1486'),
                ('VAR_1475', 'VAR_1476'),
                ('VAR_1805', 'VAR_1806'),
                ('VAR_0061', 'VAR_0062'),
                ('VAR_1649', 'VAR_1650'),
                ('VAR_0233', 'VAR_0234'),
                ('VAR_0078', 'VAR_0079'),
                ('VAR_0054', 'VAR_0055'),
                ('VAR_1770', 'VAR_1771'),
                ('VAR_1600', 'VAR_1601'),
                ('VAR_1405', 'VAR_1406'),
                ('VAR_1169', 'VAR_1170'),
                ('VAR_1811', 'VAR_1812'),
                ('VAR_1803', 'VAR_1804'),
                ('VAR_0414', 'VAR_0415'),
                ('VAR_1644', 'VAR_1645'),
                ('VAR_1772', 'VAR_1773'),
                ('VAR_0085', 'VAR_0086'),
                ('VAR_1779', 'VAR_1780'),
                ('VAR_1614', 'VAR_1615'),
                ('VAR_1607', 'VAR_1608'),
                ('VAR_1671', 'VAR_1672'),
                ('VAR_1611', 'VAR_1612'),
                ('VAR_0386', 'VAR_0387'),
                ('VAR_1279', 'VAR_1280'),
                ('VAR_1365', 'VAR_1366'),
                ('VAR_1797', 'VAR_1798'),
                ('VAR_1449', 'VAR_1450'),
                ('VAR_1310', 'VAR_1311'),
                ('VAR_0045', 'VAR_0046'),
                ('VAR_0254', 'VAR_0255'),
                ('VAR_0442', 'VAR_0443'),
                ('VAR_0509', 'VAR_0510'),
                ('VAR_0077', 'VAR_0078'),
                ('VAR_1723', 'VAR_1724'),
                ('VAR_0060', 'VAR_0061'),
                ('VAR_1443', 'VAR_1444'),
                ('VAR_1471', 'VAR_1472'),
                ('VAR_0415', 'VAR_0416'),
                ('VAR_1480', 'VAR_1481'),
                ('VAR_1926', 'VAR_1927'),
                ('VAR_0417', 'VAR_0418'),
                ('VAR_1916', 'VAR_1917'),
                ('VAR_0135', 'VAR_0136'),
                ('VAR_1909', 'VAR_1910'),
                ('VAR_0084', 'VAR_0085'),
                ('VAR_1078', 'VAR_1079'),
                ('VAR_1604', 'VAR_1605'),
                ('VAR_1520', 'VAR_1521'),
                ('VAR_0912', 'VAR_0913'),
                ('VAR_1052', 'VAR_1053'),
                ('VAR_0143', 'VAR_0144'),
                ('VAR_0119', 'VAR_0120'),
                ('VAR_1057', 'VAR_1058'),
                ('VAR_1907', 'VAR_1908'),
                ('VAR_0064', 'VAR_0065'),
                ('VAR_0429', 'VAR_0430'),
                ('VAR_0390', 'VAR_0391'),
                ('VAR_0486', 'VAR_0487'),
                ('VAR_1724', 'VAR_1725'),
                ('VAR_1288', 'VAR_1289'),
                ('VAR_1908', 'VAR_1909'),
                ('VAR_1476', 'VAR_1477'),
                ('VAR_1880', 'VAR_1881'),
                ('VAR_0379', 'VAR_0380'),
                ('VAR_1025', 'VAR_1026'),
                ('VAR_0118', 'VAR_0119'),
                ('VAR_0063', 'VAR_0064'),
                ('VAR_0127', 'VAR_0128'),
                ('VAR_1612', 'VAR_1613'),
                ('VAR_1922', 'VAR_1923'),
                ('VAR_1521', 'VAR_1522'),
                ('VAR_0532', 'VAR_0533'),
                ('VAR_0126', 'VAR_0127'),
                ('VAR_0698', 'VAR_0699'),
                ('VAR_0478', 'VAR_0479'),
                ('VAR_1091', 'VAR_1092'),
                ('VAR_1236', 'VAR_1237')]
    
    def transform_object_features(self):

        features_for_one_hot = []
        features_to_remove = []
        for i in range(0, self._train_features.shape[1]):
            feature_type = feature_utilities.identify_feature(self._train_features.iloc[:, i].append(self._test_features.iloc[:, i]))
            feature_values = self._train_features.iloc[:, i].append(self._test_features.iloc[:, i]).value_counts()

            if feature_type[0] == 'Object' and feature_type[1] == 'Date':
                self._train_features.iloc[:, i] = self._train_features.iloc[:, i].apply(feature_utilities.transform_date)
                self._test_features.iloc[:, i] = self._test_features.iloc[:, i].apply(feature_utilities.transform_date)

            # elif feature_type[0] == 'Object' and 1 < feature_values.shape[0] < 10:
            #     features_for_one_hot.append(self._train_features.columns[i])

            elif feature_values.shape[0] <= 1 or feature_values.sum() - feature_values.iloc[0] < 1000:
                print 'To Remove (not enough distinct values)', self._train_features.columns[i]
                features_to_remove.append(self._train_features.columns[i])

            elif feature_values.shape[0] > 25 * 1000:
                print 'To Remove (too many distinct values)', self._train_features.columns[i]
                features_to_remove.append(self._train_features.columns[i])

            elif feature_type[0] == 'Object':
                print 'Converting to ratio', self._train_features.columns[i]

                var = self._train_features.columns[i]
                tmp = pd.DataFrame({var: self._train_features.iloc[:, i], 'target': self._train_y['target']})
                tmp['tmp'] = 1

                tmp_pivot = pd.pivot_table(tmp, columns='target', index=var, values='tmp', aggfunc='count')

                tmp_pivot['Ratio'] = tmp_pivot.iloc[:, 1] / (tmp_pivot.iloc[:, 0] + tmp_pivot.iloc[:, 1])
                tmp_pivot['Ratio'] = tmp_pivot['Ratio'].apply(lambda x: x * (1+(np.random.random()-0.5)/100))
                tmp_pivot['Index'] = tmp_pivot.index

                filtered_pivot = tmp_pivot[(tmp_pivot.iloc[:, 0] + tmp_pivot.iloc[:, 1]) >= 30]
                self._train_features.iloc[:, i] = self._train_features.iloc[:, i].apply(feature_utilities.replace_with_avg, args=(filtered_pivot,))
                self._test_features.iloc[:, i] = self._test_features.iloc[:, i].apply(feature_utilities.replace_with_avg, args=(filtered_pivot,))

        # Remove constants / near constants
        for feature_to_remove in features_to_remove:
            self._train_features.drop(feature_to_remove, axis=1, inplace=True)
            self._test_features.drop(feature_to_remove, axis=1, inplace=True)

        # One hot part
        train_features_encoded = pd.get_dummies(self._train_features, columns=features_for_one_hot)
        test_features_encoded = pd.get_dummies(self._test_features, columns=features_for_one_hot)

        train_columns = train_features_encoded.columns.tolist()
        test_columns = test_features_encoded.columns.tolist()

        for column in (set(train_columns) - set(test_columns)):
            test_features_encoded[column] = 0

        for column in (set(test_columns) - set(train_columns)):
            train_features_encoded[column] = 0

        # Align columns
        train_columns = train_features_encoded.columns.tolist()
        train_columns.sort()

        self._train_features = train_features_encoded[train_columns]
        self._test_features = test_features_encoded[train_columns]

    def add_date_relations(self):
        for i in range(0, len(self.FULL_DATE_FEATURES)):
            for j in range(0, len(self.FULL_DATE_FEATURES)):
                if i == j:
                    continue
                feature_name = '{}_{}'.format(self.FULL_DATE_FEATURES[i], self.FULL_DATE_FEATURES[j])
                self._train_features[feature_name] = self._train_features[self.FULL_DATE_FEATURES[i]] - self._train_features[self.FULL_DATE_FEATURES[j]]
                self._test_features[feature_name] = self._test_features[self.FULL_DATE_FEATURES[i]] - self._test_features[self.FULL_DATE_FEATURES[j]]

    def transform_features(self):

        raw_train_features = self._train_features.copy()
        raw_test_features = self._test_features.copy()

        self.transform_object_features()

        for feature_name in self.FULL_DATE_FEATURES:
            # Day of week
            self._train_features[feature_name.replace('VAR', 'DOW')] = raw_train_features[feature_name].apply(feature_utilities.get_day_of_week)
            self._test_features[feature_name.replace('VAR', 'DOW')] = raw_test_features[feature_name].apply(feature_utilities.get_day_of_week)

            # Day of month
            self._train_features[feature_name.replace('VAR', 'DOM')] = raw_train_features[feature_name].apply(feature_utilities.get_day_of_month)
            self._test_features[feature_name.replace('VAR', 'DOM')] = raw_test_features[feature_name].apply(feature_utilities.get_day_of_month)

            # Day of year
            self._train_features[feature_name.replace('VAR', 'DOY')] = raw_train_features[feature_name].apply(feature_utilities.get_day_of_year)
            self._test_features[feature_name.replace('VAR', 'DOY')] = raw_test_features[feature_name].apply(feature_utilities.get_day_of_year)

            # Month
            self._train_features[feature_name.replace('VAR', 'MON')] = raw_train_features[feature_name].apply(feature_utilities.get_month)
            self._test_features[feature_name.replace('VAR', 'MON')] = raw_test_features[feature_name].apply(feature_utilities.get_month)

        # VAR_0204 parsing
        self._train_features['VAR_0204_H'] = raw_train_features['VAR_0204'].apply(feature_utilities.get_var_204_value)
        self._test_features['VAR_0204_H'] = raw_test_features['VAR_0204'].apply(feature_utilities.get_var_204_value)

        self.add_date_relations()

        # Crosses
        for cross_val in self.TO_CROSS:
            if cross_val[0] in self._train_features.columns and cross_val[1] in self._train_features.columns:
                feature_name = '{}_{}'.format(cross_val[0], cross_val[1])
                self._train_features[feature_name] = self._train_features[cross_val[0]] - self._train_features[cross_val[1]]
                self._test_features[feature_name] = self._test_features[cross_val[0]] - self._test_features[cross_val[1]]

        # ???
        self._train_features.fillna(NA_VALUE, inplace=True)
        self._test_features.fillna(NA_VALUE, inplace=True)

