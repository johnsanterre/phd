import numpy as np
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, SGDClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score,recall_score, roc_curve
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression,SGDClassifier
from joblib import Parallel, delayed
import datetime
from itertools import combinations_with_replacement
#import seaborn as sns


class Klebsiella(object):
  def __init__(self):
    self.masterLoc = '/vol/ml/jsanterre/jim/jimFinal/'
    self.kmerLoc = self.masterLoc + 'KMERS/'
    self.indexDictionaryLoc = self.masterLoc + 'index.pickle'
    self.kmer_to_col_loc = self.masterLoc + 'kmer_to_col.pickle'
    self.MLoc = self.masterLoc + '/M.npy'
    self.M_smallLoc = self.masterLoc + '/M_small.npy'
    self.logFileLoc = self.masterLoc + 'logFiles/'

    self.logFileNames=['Amikacin_MIC_SIR.log','Aztreonam.log','Cefepime.log',
                       'Cefoxitin_MIC_SIR.log','Ciprofloxacin_MIC_SIR.log',
                       'Ertapenem_KB_SIR.log','Fosfomycin_KB_SIR.log',
                       'Gentamicin_MIC_SIR.log','Imipenem_MIC_SIR.log',
                       'Levofloxacin_MIC_SIR.log','Meropenem_MIC_SIR.log',
                       'Piperacillin_Tazobactam_MIC_SIR.log',
                       'Tetracycline_MIC_SIR.log','Tigecycline_BP_SIR.log',
                       'Tobramycin_MIC_SIR.log',
                       'Trimethoprim_Sulfamethoxazole_MIC_SIR.log']
    self.antibiotics = [y.split('.')[0] for y in
                              [x.split('_')[0] for x in self.logFileNames]]
    self.isolates = ['KPN01ec', 'KPN09ec', 'KPN1000ec', 'KPN1001ec',
                     'KPN1002ec', 'KPN1004ec', 'KPN1005ec', 'KPN1007ec',
                     'KPN1010ec', 'KPN1011ec', 'KPN1012ec', 'KPN1013ec',
                     'KPN1014ec-combo', 'KPN1015ec', 'KPN1017ec', 'KPN1018ec',
                     'KPN1019ec', 'KPN1020ec', 'KPN1021ec', 'KPN1022ec',
                     'KPN1023ec', 'KPN1025ec', 'KPN1027ec', 'KPN102ec',
                     'KPN1030ec', 'KPN1031ec', 'KPN1032ec', 'KPN1033ec',
                     'KPN1034ec', 'KPN1036ec', 'KPN1037ec', 'KPN1038ec',
                     'KPN1039ec', 'KPN103ec', 'KPN1040ec', 'KPN1041ec',
                     'KPN1042ec', 'KPN1043ec', 'KPN1044ec', 'KPN1045ec',
                     'KPN1046ec', 'KPN1047ec', 'KPN1049ec-combo', 'KPN104ec',
                     'KPN1052ec-combo', 'KPN1055ec', 'KPN1056ec', 'KPN1057ec',
                     'KPN1058ec', 'KPN105ec', 'KPN1060ec', 'KPN1063ec',
                     'KPN1064ec', 'KPN1065ec', 'KPN1066ec', 'KPN1067ec',
                     'KPN1068ec-combo', 'KPN106ec', 'KPN1070ec', 'KPN1071ec',
                     'KPN1072ec-combo', 'KPN1074ec', 'KPN1075ec-combo',
                     'KPN1077ec', 'KPN1079ec-combo', 'KPN107ec',
                     'KPN1080ec-combo', 'KPN1085ec', 'KPN1086ec', 'KPN1087ec',
                     'KPN108ec', 'KPN1090ec', 'KPN1091ec', 'KPN1093ec',
                     'KPN1094ec', 'KPN1095ec', 'KPN1096ec', 'KPN1097ec',
                     'KPN1099ec', 'KPN109ec', 'KPN10ec', 'KPN1100ec',
                     'KPN1102ec', 'KPN1103ec', 'KPN1104ec', 'KPN1105ec',
                     'KPN1106ec', 'KPN1107ec', 'KPN1108ec', 'KPN1109ec',
                     'KPN110ec', 'KPN1110ec', 'KPN1111ec', 'KPN1113ec',
                     'KPN1115ec', 'KPN1116ec', 'KPN1117ec', 'KPN1118ec',
                     'KPN1119ec', 'KPN111ec', 'KPN1120ec', 'KPN1122ec',
                     'KPN1123ec', 'KPN1124ec', 'KPN1125ec', 'KPN1126ec',
                     'KPN1127ec', 'KPN1128ec', 'KPN1129ec', 'KPN112ec',
                     'KPN1131ec', 'KPN1132ec', 'KPN1133ec', 'KPN1134ec',
                     'KPN1135ec', 'KPN1136ec', 'KPN1137ec', 'KPN1139ec',
                     'KPN113ec', 'KPN1140ec', 'KPN1141ec', 'KPN1142ec',
                     'KPN1143ec-combo', 'KPN1145ec', 'KPN1146ec', 'KPN1147ec',
                     'KPN1148ec', 'KPN1149ec', 'KPN114ec', 'KPN1150ec-combo',
                     'KPN1151ec', 'KPN1152ec', 'KPN1153ec', 'KPN1154ec',
                     'KPN1155ec', 'KPN1156ec', 'KPN1158ec', 'KPN115ec',
                     'KPN1160ec', 'KPN1162ec', 'KPN1163ec', 'KPN1164ec',
                     'KPN1165ec', 'KPN1166ec', 'KPN1167ec', 'KPN1168ec',
                     'KPN1169ec', 'KPN116ec', 'KPN1170ec', 'KPN1171ec',
                     'KPN1172ec', 'KPN1173ec', 'KPN1174ec', 'KPN1176ec',
                     'KPN1177ec', 'KPN1178ec', 'KPN1179ec', 'KPN117ec',
                     'KPN1181ec', 'KPN1182ec', 'KPN1183ec', 'KPN1184ec',
                     'KPN1185ec', 'KPN1186ec', 'KPN1187ec-combo', 'KPN1188ec',
                     'KPN1189ec', 'KPN118ec', 'KPN1190ec', 'KPN1191ec',
                     'KPN1192ec', 'KPN1193ec', 'KPN1194ec', 'KPN1195ec',
                     'KPN1196ec', 'KPN1197ec', 'KPN1198ec', 'KPN1199ec-combo',
                     'KPN119ec', 'KPN11ec', 'KPN1200ec', 'KPN1202ec',
                     'KPN1203ec', 'KPN1204ec', 'KPN1205ec', 'KPN1206ec',
                     'KPN1207ec', 'KPN1208ec', 'KPN1209ec-combo', 'KPN120ec',
                     'KPN1210ec', 'KPN1213ec', 'KPN1214ec', 'KPN1216ec',
                     'KPN1217ec', 'KPN1218ec', 'KPN1219ec', 'KPN121ec',
                     'KPN1220ec', 'KPN1221ec', 'KPN1222ec-combo', 'KPN1224ec',
                     'KPN1226ec-combo', 'KPN122ec', 'KPN1230ec', 'KPN1233ec',
                     'KPN1239ec', 'KPN123ec', 'KPN1240ec', 'KPN1241ec',
                     'KPN1242ec-combo', 'KPN1243ec', 'KPN1244ec', 'KPN1245ec',
                     'KPN1246ec', 'KPN1248ec', 'KPN1249ec', 'KPN124ec',
                     'KPN1251ec', 'KPN1252ec', 'KPN1254ec', 'KPN1255ec',
                     'KPN1256ec', 'KPN1257ec-combo', 'KPN1259ec-combo',
                     'KPN125ec', 'KPN1260ec', 'KPN1261ec', 'KPN1262ec',
                     'KPN1263ec', 'KPN1264ec', 'KPN1265ec', 'KPN1266ec',
                     'KPN1267ec', 'KPN1268ec', 'KPN1269ec', 'KPN126ec',
                     'KPN1271ec', 'KPN1272ec', 'KPN1273ec', 'KPN1274ec',
                     'KPN1275ec', 'KPN1276ec', 'KPN1277ec', 'KPN1278ec',
                     'KPN1279ec', 'KPN1280ec-combo', 'KPN1281ec', 'KPN1282ec',
                     'KPN1283ec', 'KPN1284ec', 'KPN1285ec', 'KPN1286ec',
                     'KPN128ec', 'KPN129ec', 'KPN12ec', 'KPN1301ec',
                     'KPN1302ec', 'KPN1303ec', 'KPN1304ec', 'KPN1305ec',
                     'KPN1306ec', 'KPN1307ec', 'KPN1308ec', 'KPN1309ec',
                     'KPN130ec', 'KPN1311ec', 'KPN1312ec', 'KPN1314ec',
                     'KPN1315ec', 'KPN1316ec', 'KPN1317ec', 'KPN1318ec',
                     'KPN1319ec', 'KPN131ec', 'KPN1320ec', 'KPN1321ec',
                     'KPN1322ec', 'KPN1323ec', 'KPN1325ec', 'KPN1326ec',
                     'KPN1327ec', 'KPN1328ec', 'KPN1329ec', 'KPN132ec',
                     'KPN1330ec', 'KPN1331ec', 'KPN1332ec', 'KPN1333ec',
                     'KPN1334ec', 'KPN1335ec', 'KPN1336ec', 'KPN1337ec',
                     'KPN1338ec', 'KPN1339ec', 'KPN133ec', 'KPN1340ec',
                     'KPN1341ec', 'KPN1342ec', 'KPN1344ec', 'KPN1346ec',
                     'KPN1347ec', 'KPN1348ec', 'KPN1349ec', 'KPN134ec',
                     'KPN1350ec', 'KPN1351ec', 'KPN1352ec', 'KPN1353ec',
                     'KPN1354ec', 'KPN1355ec', 'KPN1356ec', 'KPN1357ec',
                     'KPN1358ec', 'KPN1359ec', 'KPN135ec', 'KPN1360ec',
                     'KPN1361ec', 'KPN1362ec', 'KPN1363ec', 'KPN1364ec',
                     'KPN1365ec', 'KPN1366ec', 'KPN136ec', 'KPN1370ec',
                     'KPN1371ec', 'KPN1372ec', 'KPN1373ec', 'KPN1374ec-combo',
                     'KPN1375ec', 'KPN1376ec', 'KPN1377ec', 'KPN1378ec',
                     'KPN137ec', 'KPN1380ec', 'KPN1381ec', 'KPN1382ec',
                     'KPN1383ec', 'KPN1385ec', 'KPN1386ec', 'KPN1387ec',
                     'KPN1388ec', 'KPN1389ec', 'KPN138ec', 'KPN1390ec',
                     'KPN1393ec', 'KPN1394ec', 'KPN1397ec', 'KPN139ec',
                     'KPN13ec', 'KPN1400ec', 'KPN1401ec', 'KPN1402ec',
                     'KPN1403ec', 'KPN1404ec', 'KPN1405ec', 'KPN1406ec',
                     'KPN1407ec', 'KPN1408ec', 'KPN1409ec', 'KPN140ec',
                     'KPN1410ec', 'KPN1411ec', 'KPN1412ec', 'KPN1413ec',
                     'KPN1414ec', 'KPN1415ec', 'KPN1416ec', 'KPN1417ec',
                     'KPN1418ec', 'KPN141ec', 'KPN1420ec', 'KPN1421ec',
                     'KPN1422ec', 'KPN1423ec', 'KPN1424ec', 'KPN1425ec',
                     'KPN1426ec', 'KPN1427ec', 'KPN1428ec', 'KPN1429ec',
                     'KPN142ec', 'KPN1430ec', 'KPN1431ec', 'KPN1432ec',
                     'KPN1433ec', 'KPN1434ec', 'KPN1435ec', 'KPN1436ec',
                     'KPN1437ec', 'KPN1438ec', 'KPN1439ec', 'KPN143ec',
                     'KPN1440ec', 'KPN1441ec', 'KPN1442ec', 'KPN1443ec',
                     'KPN1444ec', 'KPN1445ec', 'KPN1446ec', 'KPN1447ec',
                     'KPN1448ec', 'KPN1449ec', 'KPN144ec', 'KPN1450ec',
                     'KPN1451ec', 'KPN1452ec', 'KPN1453ec', 'KPN1454ec',
                     'KPN1455ec', 'KPN1456ec', 'KPN1457ec', 'KPN1458ec',
                     'KPN145ec', 'KPN1460ec', 'KPN1461ec', 'KPN1462ec',
                     'KPN1463ec', 'KPN1464ec', 'KPN1465ec', 'KPN1466ec',
                     'KPN1467ec', 'KPN1469ec', 'KPN1471ec', 'KPN1472ec',
                     'KPN1473ec', 'KPN1474ec', 'KPN1475ec', 'KPN1476ec',
                     'KPN1477ec', 'KPN1478ec', 'KPN147ec', 'KPN1480ec',
                     'KPN1481ec', 'KPN1482ec', 'KPN1483ec', 'KPN1484ec',
                     'KPN1485ec', 'KPN1486ec', 'KPN1487ec', 'KPN1488ec',
                     'KPN1489ec', 'KPN148ec', 'KPN1490ec', 'KPN1491ec',
                     'KPN1493ec', 'KPN1494ec', 'KPN1495ec', 'KPN1496ec',
                     'KPN1497ec', 'KPN1498ec', 'KPN1499ec', 'KPN149ec',
                     'KPN14ec', 'KPN1500ec', 'KPN1501ec', 'KPN1502ec',
                     'KPN1504ec', 'KPN1505ec', 'KPN1506ec', 'KPN1507ec',
                     'KPN1508ec', 'KPN1509ec', 'KPN150ec', 'KPN1510ec',
                     'KPN1511ec', 'KPN1512ec', 'KPN1513ec', 'KPN1514ec',
                     'KPN1515ec', 'KPN1516ec', 'KPN1517ec', 'KPN1518ec',
                     'KPN151ec', 'KPN1520ec', 'KPN1521ec', 'KPN1522ec',
                     'KPN1523ec', 'KPN1524ec', 'KPN1525ec', 'KPN1526ec',
                     'KPN1527ec', 'KPN1529ec', 'KPN1530ec', 'KPN1531ec',
                     'KPN1532ec', 'KPN1533ec', 'KPN1534ec', 'KPN1535ec',
                     'KPN1536ec', 'KPN1537ec', 'KPN1539ec', 'KPN1540ec',
                     'KPN1542ec', 'KPN1543ec', 'KPN1544ec', 'KPN1545ec',
                     'KPN1546ec', 'KPN1547ec', 'KPN1548ec', 'KPN1549ec',
                     'KPN154ec', 'KPN1550ec', 'KPN1551ec', 'KPN1552ec',
                     'KPN1555ec', 'KPN1556ec', 'KPN1557ec', 'KPN1558ec',
                     'KPN1559ec', 'KPN155ec', 'KPN1560ec', 'KPN1561ec',
                     'KPN1562ec', 'KPN1564ec', 'KPN1565ec', 'KPN1566ec',
                     'KPN1567ec', 'KPN1568ec', 'KPN1569ec', 'KPN1570ec',
                     'KPN1571ec', 'KPN1572ec', 'KPN1573ec', 'KPN1575ec',
                     'KPN1576ec', 'KPN1577ec', 'KPN1578ec', 'KPN1579ec',
                     'KPN157ec', 'KPN1580ec', 'KPN1581ec', 'KPN1582ec',
                     'KPN1583ec', 'KPN1584ec', 'KPN1585ec', 'KPN1586ec',
                     'KPN1587ec', 'KPN1588ec', 'KPN1590ec', 'KPN1591ec',
                     'KPN1592ec', 'KPN1593ec', 'KPN1594ec', 'KPN1595ec',
                     'KPN1596ec', 'KPN1597ec', 'KPN1598ec', 'KPN159ec',
                     'KPN15ec', 'KPN1603ec', 'KPN1604ec', 'KPN1605ec',
                     'KPN1607ec', 'KPN1608ec', 'KPN1609ec', 'KPN160ec',
                     'KPN1610ec', 'KPN1611ec', 'KPN1612ec', 'KPN1613ec',
                     'KPN1616ec', 'KPN1617ec', 'KPN1618ec', 'KPN161ec',
                     'KPN1621ec', 'KPN1623ec', 'KPN1624ec', 'KPN1625ec',
                     'KPN1626ec', 'KPN1627ec', 'KPN1629ec', 'KPN162ec',
                     'KPN1634ec', 'KPN1635ec', 'KPN1639ec', 'KPN163ec',
                     'KPN1642ec', 'KPN1643ec', 'KPN1644ec', 'KPN1645ec',
                     'KPN1646ec', 'KPN1647ec', 'KPN1648ec', 'KPN1649ec',
                     'KPN164ec', 'KPN1650ec', 'KPN1651ec', 'KPN1652ec',
                     'KPN1653ec', 'KPN1654ec', 'KPN1655ec', 'KPN1656ec',
                     'KPN1657ec', 'KPN1658ec', 'KPN1659ec', 'KPN1660ec',
                     'KPN1662ec', 'KPN1663ec', 'KPN1664ec', 'KPN1665ec',
                     'KPN1666ec', 'KPN1667ec', 'KPN1668ec', 'KPN1669ec',
                     'KPN166ec', 'KPN1670ec', 'KPN1671ec', 'KPN1672ec',
                     'KPN1673ec', 'KPN1674ec', 'KPN1675ec', 'KPN1676ec',
                     'KPN1677ec', 'KPN1678ec', 'KPN1679ec', 'KPN167ec',
                     'KPN1680ec', 'KPN1682ec', 'KPN1684ec', 'KPN1685ec',
                     'KPN1686ec', 'KPN1687ec', 'KPN1688ec', 'KPN1689ec',
                     'KPN1690ec', 'KPN1691ec', 'KPN1692ec', 'KPN1693ec',
                     'KPN1694ec', 'KPN1695ec', 'KPN1696ec', 'KPN1697ec',
                     'KPN1698ec', 'KPN1699ec', 'KPN169ec', 'KPN1700ec',
                     'KPN1701ec', 'KPN1703ec', 'KPN1704ec', 'KPN1705ec',
                     'KPN1706ec', 'KPN1707ec', 'KPN1708ec', 'KPN1709ec',
                     'KPN1710ec', 'KPN1711ec', 'KPN1712ec', 'KPN1713ec',
                     'KPN1714ec', 'KPN1715ec', 'KPN1716ec', 'KPN1717ec',
                     'KPN1718ec', 'KPN1719ec', 'KPN1720ec', 'KPN1721ec',
                     'KPN1722ec', 'KPN1723ec', 'KPN1724ec', 'KPN1725ec',
                     'KPN1726ec', 'KPN1727ec', 'KPN1728ec', 'KPN1729ec',
                     'KPN172ec', 'KPN1730ec', 'KPN1734ec', 'KPN1735ec',
                     'KPN1736ec', 'KPN1737ec', 'KPN1738ec', 'KPN1739ec',
                     'KPN1740ec', 'KPN1741ec', 'KPN1743ec', 'KPN1744ec',
                     'KPN1745ec', 'KPN1746ec', 'KPN1747ec', 'KPN1748ec',
                     'KPN1749ec', 'KPN174ec', 'KPN1751ec', 'KPN1752ec',
                     'KPN1754ec', 'KPN1755ec', 'KPN1756ec', 'KPN1757ec',
                     'KPN1759ec', 'KPN1760ec', 'KPN1761ec', 'KPN1762ec',
                     'KPN1763ec', 'KPN1764ec', 'KPN1765ec', 'KPN1766ec',
                     'KPN1767ec', 'KPN1769ec', 'KPN176ec', 'KPN1770ec',
                     'KPN1771ec', 'KPN1772ec', 'KPN1774ec', 'KPN1775ec',
                     'KPN1776ec', 'KPN1777ec', 'KPN1778ec', 'KPN1779ec',
                     'KPN177ec', 'KPN1781ec', 'KPN1782ec', 'KPN1783ec',
                     'KPN1784ec', 'KPN1785ec', 'KPN1786ec', 'KPN1788ec',
                     'KPN1789ec', 'KPN178ec', 'KPN179ec', 'KPN17ec',
                     'KPN180ec', 'KPN1817ec', 'KPN1819ec', 'KPN181ec',
                     'KPN1820ec', 'KPN1824ec', 'KPN1826ec', 'KPN1827ec',
                     'KPN1828ec', 'KPN1829ec', 'KPN182ec', 'KPN1831ec',
                     'KPN1832ec', 'KPN1835ec', 'KPN1836ec', 'KPN1838ec',
                     'KPN1839ec', 'KPN183ec', 'KPN1840ec', 'KPN1841ec',
                     'KPN1842ec', 'KPN1843ec', 'KPN1844ec', 'KPN1845ec',
                     'KPN1847ec', 'KPN1848ec', 'KPN1849ec', 'KPN184ec',
                     'KPN1850ec', 'KPN1851ec', 'KPN1852ec', 'KPN1853ec',
                     'KPN1854ec', 'KPN1855ec', 'KPN1856ec', 'KPN1857ec',
                     'KPN1859ec', 'KPN185ec', 'KPN1860ec', 'KPN1862ec',
                     'KPN1863ec', 'KPN1864ec', 'KPN1866ec', 'KPN1867ec',
                     'KPN1868ec', 'KPN186ec', 'KPN1873ec', 'KPN1875ec',
                     'KPN1876ec', 'KPN1877ec', 'KPN1879ec', 'KPN187ec',
                     'KPN1880ec', 'KPN1881ec', 'KPN1882ec', 'KPN1883ec',
                     'KPN1884ec', 'KPN1887ec', 'KPN1888ec', 'KPN1889ec',
                     'KPN188ec', 'KPN1890ec', 'KPN1891ec', 'KPN1893ec',
                     'KPN1894ec', 'KPN1895ec', 'KPN1897ec', 'KPN1898ec',
                     'KPN189ec', 'KPN18ec', 'KPN1900ec', 'KPN1901ec',
                     'KPN1902ec', 'KPN1903ec', 'KPN1904ec', 'KPN1905ec',
                     'KPN1906ec', 'KPN190ec', 'KPN1911ec', 'KPN1912ec',
                     'KPN1913ec', 'KPN1914ec', 'KPN1917ec', 'KPN1918ec',
                     'KPN191ec', 'KPN1920ec', 'KPN1921ec', 'KPN1924ec',
                     'KPN1925ec', 'KPN1926ec', 'KPN1927ec', 'KPN1928ec',
                     'KPN1929ec', 'KPN192ec', 'KPN1930ec', 'KPN1931ec',
                     'KPN1932ec', 'KPN1933ec', 'KPN1934ec', 'KPN1936ec',
                     'KPN1937ec', 'KPN1938ec', 'KPN1939ec', 'KPN1940ec',
                     'KPN1942ec', 'KPN1943ec', 'KPN1945ec-combo',
                     'KPN1946ec-combo', 'KPN1947ec', 'KPN1948cec', 'KPN1949ec',
                     'KPN194ec', 'KPN1950ec-combo', 'KPN1951ec-combo',
                     'KPN1952ec', 'KPN1953ec', 'KPN1954ec', 'KPN1955ec',
                     'KPN1956ec', 'KPN1958ec', 'KPN195ec-combo', 'KPN1960ec',
                     'KPN1961ec', 'KPN1962ec', 'KPN1963ec', 'KPN1968ec',
                     'KPN196ec', 'KPN1973ec', 'KPN1974ec', 'KPN1975ec',
                     'KPN1976ec', 'KPN1977ec', 'KPN197ec', 'KPN1980ec',
                     'KPN1981ec', 'KPN1982ec', 'KPN1983ec', 'KPN1984ec',
                     'KPN1985ec', 'KPN1986ec', 'KPN1987ec', 'KPN1988ec',
                     'KPN1989ec', 'KPN198ec', 'KPN1991ec', 'KPN1992ec',
                     'KPN1993ec', 'KPN1994ec', 'KPN1995ec', 'KPN1997ec',
                     'KPN1998ec', 'KPN1999ec', 'KPN199ec', 'KPN19ec',
                     'KPN2000ec', 'KPN2001ec', 'KPN2002ec', 'KPN2003ec',
                     'KPN2004ec', 'KPN2006ec', 'KPN2007ec', 'KPN2008ec',
                     'KPN2009ec', 'KPN200ec', 'KPN2010ec', 'KPN2011ec',
                     'KPN2012ec', 'KPN2013ec', 'KPN2015ec', 'KPN2016ec',
                     'KPN2017ec', 'KPN2018ec', 'KPN2019ec', 'KPN201ec',
                     'KPN2020ec', 'KPN2021ec', 'KPN2022ec', 'KPN2023ec',
                     'KPN2024ec', 'KPN2026ec', 'KPN2027ec', 'KPN2028ec',
                     'KPN2029ec', 'KPN202ec', 'KPN2030ec', 'KPN2031ec',
                     'KPN2032ec', 'KPN2034ec', 'KPN2035ec', 'KPN2036ec',
                     'KPN2037ec', 'KPN2038ec', 'KPN2039ec', 'KPN2040ec',
                     'KPN2041ec', 'KPN2042ec', 'KPN2043ec', 'KPN2044ec',
                     'KPN2045ec', 'KPN2046ec', 'KPN2047ec', 'KPN2048ec',
                     'KPN2049ec', 'KPN204ec', 'KPN2050ec', 'KPN2051ec',
                     'KPN2052ec', 'KPN2053ec', 'KPN2054ec', 'KPN2055ec',
                     'KPN2056ec', 'KPN2057ec', 'KPN2058ec', 'KPN205ec',
                     'KPN2060ec', 'KPN2061ec', 'KPN2062ec', 'KPN2063ec',
                     'KPN2064ec', 'KPN2065ec', 'KPN2066ec', 'KPN2067ec',
                     'KPN2069ec', 'KPN206ec', 'KPN2070ec', 'KPN2071ec',
                     'KPN2072ec', 'KPN2073ec', 'KPN2075ec', 'KPN2076ec',
                     'KPN2077ec', 'KPN2078ec', 'KPN2079ec', 'KPN207ec',
                     'KPN2080ec', 'KPN2081ec', 'KPN2082ec', 'KPN2083ec',
                     'KPN2084ec', 'KPN2085ec', 'KPN2086ec', 'KPN2087ec',
                     'KPN2088ec', 'KPN208ec', 'KPN2090ec', 'KPN2091ec',
                     'KPN2092ec', 'KPN2093ec', 'KPN2094ec', 'KPN2095ec',
                     'KPN2096ec', 'KPN2097ec', 'KPN2098ec', 'KPN2099ec',
                     'KPN209ec', 'KPN20ec', 'KPN2100ec', 'KPN2101ec',
                     'KPN2103ec', 'KPN2104ec', 'KPN2105ec', 'KPN2106ec',
                     'KPN2108ec', 'KPN2109ec', 'KPN210ec', 'KPN2111ec',
                     'KPN2113ec', 'KPN2114ec', 'KPN2115ec', 'KPN2116ec',
                     'KPN2117ec', 'KPN2118ec', 'KPN2121ec', 'KPN2122ec',
                     'KPN2123ec', 'KPN2124ec', 'KPN2125ec', 'KPN2126ec',
                     'KPN2127ec', 'KPN2128ec', 'KPN2129ec', 'KPN212ec',
                     'KPN2130ec', 'KPN213ec', 'KPN2149ec', 'KPN214ec',
                     'KPN2150ec', 'KPN2151ec', 'KPN2152ec', 'KPN2154ec',
                     'KPN2155ec', 'KPN2156ec', 'KPN2157ec', 'KPN2158ec',
                     'KPN2159ec', 'KPN215ec', 'KPN2160ec', 'KPN2161ec',
                     'KPN2162ec', 'KPN2163ec', 'KPN2164ec', 'KPN2166ec',
                     'KPN2167ec', 'KPN2168ec', 'KPN2169ec', 'KPN216ec',
                     'KPN2170ec', 'KPN2171ec', 'KPN2172ec', 'KPN2173ec',
                     'KPN2174ec', 'KPN217ec', 'KPN218ec', 'KPN219ec',
                     'KPN220ec', 'KPN221ec', 'KPN222ec', 'KPN223ec',
                     'KPN224ec', 'KPN226ec', 'KPN227ec', 'KPN228ec',
                     'KPN229ec', 'KPN22ec', 'KPN230ec', 'KPN231ec', 'KPN232ec',
                     'KPN233ec', 'KPN234ec', 'KPN235ec', 'KPN236ec',
                     'KPN237ec', 'KPN238ec', 'KPN239ec', 'KPN23ec', 'KPN240ec',
                     'KPN241ec', 'KPN242ec', 'KPN244ec', 'KPN245ec',
                     'KPN246ec', 'KPN247ec', 'KPN248ec', 'KPN249ec', 'KPN24ec',
                     'KPN250ec', 'KPN251ec', 'KPN252ec', 'KPN254ec',
                     'KPN256ec', 'KPN257ec', 'KPN258ec', 'KPN259ec', 'KPN25ec',
                     'KPN260ec', 'KPN261ec', 'KPN262ec', 'KPN263ec',
                     'KPN264ec', 'KPN265ec', 'KPN267ec', 'KPN268ec',
                     'KPN269ec', 'KPN26ec', 'KPN270ec', 'KPN271ec', 'KPN272ec',
                     'KPN273ec', 'KPN274ec', 'KPN275ec', 'KPN276ec',
                     'KPN277ec', 'KPN278ec', 'KPN279ec', 'KPN27ec', 'KPN280ec',
                     'KPN281ec', 'KPN282ec', 'KPN283ec', 'KPN284ec',
                     'KPN285ec', 'KPN286ec', 'KPN287ec', 'KPN288ec',
                     'KPN289ec', 'KPN28ec', 'KPN290ec', 'KPN291ec', 'KPN292ec',
                     'KPN293ec', 'KPN294ec', 'KPN295ec', 'KPN296ec',
                     'KPN297ec', 'KPN299ec', 'KPN29ec', 'KPN2ec', 'KPN300ec',
                     'KPN301ec', 'KPN302ec', 'KPN303ec', 'KPN304ec',
                     'KPN305ec', 'KPN306ec', 'KPN307ec', 'KPN308ec',
                     'KPN309ec', 'KPN30ec', 'KPN310ec', 'KPN311ec', 'KPN312ec',
                     'KPN313ec', 'KPN314ec', 'KPN315ec', 'KPN316ec',
                     'KPN317ec', 'KPN318ec', 'KPN319ec', 'KPN31ec', 'KPN320ec',
                     'KPN321ec', 'KPN322ec', 'KPN323ec', 'KPN324ec',
                     'KPN325ec', 'KPN327ec', 'KPN328ec', 'KPN329ec', 'KPN32ec',
                     'KPN330ec', 'KPN331ec', 'KPN332ec', 'KPN333ec',
                     'KPN334ec', 'KPN335ec', 'KPN336ec', 'KPN337ec',
                     'KPN338ec', 'KPN339ec', 'KPN33ec', 'KPN340ec', 'KPN342ec',
                     'KPN343ec', 'KPN345ec', 'KPN346ec', 'KPN347ec',
                     'KPN348ec', 'KPN349ec', 'KPN34ec', 'KPN350ec', 'KPN353ec',
                     'KPN354ec', 'KPN355ec', 'KPN356ec', 'KPN357ec',
                     'KPN358ec', 'KPN359ec', 'KPN360ec', 'KPN361ec',
                     'KPN362ec', 'KPN363ec', 'KPN364ec', 'KPN365ec-combo',
                     'KPN366ec', 'KPN367ec', 'KPN368ec', 'KPN369ec', 'KPN36ec',
                     'KPN370ec', 'KPN371ec', 'KPN372ec', 'KPN373ec',
                     'KPN374ec', 'KPN375ec', 'KPN376ec', 'KPN377ec',
                     'KPN378ec', 'KPN379ec', 'KPN37ec', 'KPN380ec', 'KPN381ec',
                     'KPN382ec', 'KPN383ec', 'KPN384ec', 'KPN385ec',
                     'KPN386ec', 'KPN387ec', 'KPN388ec', 'KPN389ec', 'KPN38ec',
                     'KPN390ec', 'KPN392ec', 'KPN393ec', 'KPN394ec',
                     'KPN395ec', 'KPN396ec', 'KPN397ec', 'KPN398ec',
                     'KPN399ec', 'KPN3ec', 'KPN400ec', 'KPN401ec', 'KPN402ec',
                     'KPN403ec', 'KPN404ec', 'KPN405ec', 'KPN406ec',
                     'KPN407ec', 'KPN408ec', 'KPN409ec', 'KPN411ec',
                     'KPN412ec', 'KPN413ec', 'KPN414ec', 'KPN415ec',
                     'KPN416ec', 'KPN417ec', 'KPN418ec', 'KPN419ec',
                     'KPN41ec-combo', 'KPN420ec', 'KPN421ec', 'KPN422ec',
                     'KPN423ec', 'KPN424ec', 'KPN425ec', 'KPN426ec',
                     'KPN427ec', 'KPN428ec', 'KPN429ec', 'KPN42ec-combo',
                     'KPN430ec', 'KPN431ec', 'KPN432ec', 'KPN434ec',
                     'KPN435ec', 'KPN436ec', 'KPN437ec', 'KPN438ec',
                     'KPN439ec', 'KPN43ec-combo', 'KPN440ec', 'KPN441ec',
                     'KPN442ec', 'KPN443ec', 'KPN444ec', 'KPN445ec',
                     'KPN446ec', 'KPN447ec', 'KPN448ec', 'KPN449ec',
                     'KPN44ec-combo', 'KPN450ec', 'KPN451ec', 'KPN453ec',
                     'KPN455ec', 'KPN457ec', 'KPN458ec', 'KPN460ec',
                     'KPN461ec', 'KPN462ec', 'KPN463ec', 'KPN464ec',
                     'KPN465ec', 'KPN466ec', 'KPN467ec', 'KPN468ec',
                     'KPN469ec', 'KPN470ec', 'KPN471ec', 'KPN472ec',
                     'KPN473ec', 'KPN475ec', 'KPN477ec', 'KPN479ec',
                     'KPN480ec', 'KPN481ec', 'KPN482ec', 'KPN483ec',
                     'KPN484ec', 'KPN485ec', 'KPN486ec', 'KPN4ec',
                     'KPN503ec-combo', 'KPN504ec-combo', 'KPN507ec-combo',
                     'KPN508ec-combo', 'KPN509ec-combo', 'KPN50ec-combo',
                     'KPN510ec-combo', 'KPN511ec-combo', 'KPN512ec-combo',
                     'KPN513ec-combo', 'KPN514ec-combo', 'KPN515ec-combo',
                     'KPN516ec-combo', 'KPN517ec-combo', 'KPN520ec-combo',
                     'KPN521ec-combo', 'KPN522ec-combo', 'KPN523ec-combo',
                     'KPN524ec-combo', 'KPN525ec-combo', 'KPN526ec',
                     'KPN527ec', 'KPN528ec', 'KPN530ec', 'KPN532ec',
                     'KPN533ec', 'KPN534ec', 'KPN535ec-combo', 'KPN536ec',
                     'KPN537ec', 'KPN538ec', 'KPN539ec', 'KPN53ec-combo',
                     'KPN540ec', 'KPN541ec', 'KPN542ec', 'KPN544ec',
                     'KPN545ec', 'KPN546ec', 'KPN547ec-combo', 'KPN548ec',
                     'KPN550ec', 'KPN551ec', 'KPN552ec', 'KPN553ec',
                     'KPN554ec', 'KPN555ec', 'KPN556ec', 'KPN557ec',
                     'KPN558ec', 'KPN559ec-combo', 'KPN560ec', 'KPN561ec',
                     'KPN563ec', 'KPN564ec', 'KPN566ec', 'KPN567ec',
                     'KPN568ec', 'KPN569ec', 'KPN571ec-combo', 'KPN572ec',
                     'KPN573ec-combo', 'KPN574ec', 'KPN575ec', 'KPN576ec',
                     'KPN577ec', 'KPN578ec', 'KPN579ec', 'KPN57ec-combo',
                     'KPN580ec', 'KPN581ec', 'KPN582ec-combo',
                     'KPN583ec-combo', 'KPN584ec-combo', 'KPN585ec-combo',
                     'KPN587ec', 'KPN588ec', 'KPN590ec', 'KPN591ec',
                     'KPN592ec', 'KPN593ec', 'KPN594ec', 'KPN596ec',
                     'KPN598ec', 'KPN599ec', 'KPN5ec-combo', 'KPN600ec',
                     'KPN601ec', 'KPN603ec', 'KPN604ec', 'KPN605ec',
                     'KPN606ec', 'KPN608ec', 'KPN609ec-combo', 'KPN610ec',
                     'KPN611ec', 'KPN612ec', 'KPN613ec', 'KPN614ec',
                     'KPN615ec', 'KPN616ec', 'KPN617ec', 'KPN618ec',
                     'KPN619ec-combo', 'KPN620ec', 'KPN621ec',
                     'KPN622ec-combo', 'KPN623ec-combo', 'KPN626ec-combo',
                     'KPN628ec-combo', 'KPN629ec-combo', 'KPN630ec-combo',
                     'KPN631ec-combo', 'KPN632ec-combo', 'KPN635ec-combo',
                     'KPN636ec-combo', 'KPN639ec-combo', 'KPN63ec-combo',
                     'KPN641ec-combo', 'KPN648ec-combo', 'KPN649ec-combo',
                     'KPN658ec-combo', 'KPN65ec-combo', 'KPN660ec-combo',
                     'KPN667ec-combo', 'KPN669ec-combo', 'KPN670ec-combo',
                     'KPN674ec-combo', 'KPN675ec', 'KPN676ec', 'KPN677ec',
                     'KPN678ec', 'KPN679ec', 'KPN680ec', 'KPN681ec',
                     'KPN682ec', 'KPN683ec-combo', 'KPN684ec',
                     'KPN685ec-combo', 'KPN686ec-combo', 'KPN687ec-combo',
                     'KPN688ec', 'KPN689ec', 'KPN690ec', 'KPN691ec',
                     'KPN692ec-combo', 'KPN693ec', 'KPN694ec', 'KPN695ec',
                     'KPN696ec', 'KPN697ec', 'KPN698ec-combo', 'KPN6ec',
                     'KPN700ec', 'KPN701ec', 'KPN703ec', 'KPN704ec-combo',
                     'KPN705ec', 'KPN707ec', 'KPN708ec', 'KPN709ec',
                     'KPN711ec-combo', 'KPN712ec', 'KPN713ec', 'KPN714ec',
                     'KPN715ec', 'KPN716ec', 'KPN717ec', 'KPN718ec',
                     'KPN719ec', 'KPN720ec', 'KPN721ec-combo',
                     'KPN722ec-combo', 'KPN723ec-combo', 'KPN724ec',
                     'KPN725ec', 'KPN727ec', 'KPN728ec-combo', 'KPN729ec',
                     'KPN731ec', 'KPN733ec', 'KPN734ec', 'KPN735ec-combo',
                     'KPN736ec', 'KPN737ec', 'KPN738ec', 'KPN740ec',
                     'KPN741ec', 'KPN742ec', 'KPN744ec', 'KPN745ec',
                     'KPN747ec', 'KPN748ec', 'KPN749ec', 'KPN751ec',
                     'KPN752ec', 'KPN753ec', 'KPN754ec', 'KPN755ec',
                     'KPN756ec', 'KPN758ec', 'KPN759ec', 'KPN760ec',
                     'KPN762ec', 'KPN763ec', 'KPN765ec', 'KPN766ec',
                     'KPN768ec', 'KPN769ec', 'KPN770ec', 'KPN771ec-combo',
                     'KPN772ec', 'KPN773ec', 'KPN775ec', 'KPN777ec',
                     'KPN778ec', 'KPN780ec', 'KPN781ec', 'KPN782ec-combo',
                     'KPN783ec', 'KPN785ec', 'KPN786ec', 'KPN788ec',
                     'KPN790ec', 'KPN792ec-combo', 'KPN793ec', 'KPN795ec',
                     'KPN796ec', 'KPN797ec', 'KPN799ec', 'KPN7ec', 'KPN800ec',
                     'KPN801ec', 'KPN802ec', 'KPN803ec', 'KPN807ec',
                     'KPN809ec', 'KPN811ec-combo', 'KPN813ec', 'KPN814ec',
                     'KPN816ec', 'KPN817ec', 'KPN818ec', 'KPN819ec',
                     'KPN821ec', 'KPN822ec', 'KPN823ec-combo', 'KPN824ec',
                     'KPN825ec-combo', 'KPN826ec', 'KPN828ec', 'KPN829ec',
                     'KPN830ec', 'KPN831ec', 'KPN832ec', 'KPN835ec',
                     'KPN836ec', 'KPN837ec', 'KPN838ec', 'KPN839ec',
                     'KPN840ec', 'KPN841ec', 'KPN843ec', 'KPN844ec',
                     'KPN848ec-combo', 'KPN849ec', 'KPN850ec', 'KPN852ec',
                     'KPN853ec', 'KPN854ec', 'KPN857ec', 'KPN858ec',
                     'KPN860ec-combo', 'KPN863ec', 'KPN864ec', 'KPN866ec',
                     'KPN867ec', 'KPN868ec-combo', 'KPN870ec-combo',
                     'KPN871ec', 'KPN874ec', 'KPN875ec', 'KPN876ec',
                     'KPN877ec', 'KPN878ec', 'KPN879ec', 'KPN880ec',
                     'KPN881ec', 'KPN882ec', 'KPN883ec', 'KPN886ec',
                     'KPN887ec', 'KPN888ec', 'KPN890ec', 'KPN891ec',
                     'KPN892ec', 'KPN893ec', 'KPN894ec', 'KPN895ec',
                     'KPN896ec', 'KPN899ec', 'KPN8ec', 'KPN900ec', 'KPN902ec',
                     'KPN903ec', 'KPN904ec', 'KPN906ec', 'KPN907ec',
                     'KPN908ec', 'KPN909ec', 'KPN911ec', 'KPN912ec',
                     'KPN913ec', 'KPN914ec', 'KPN915ec', 'KPN916ec',
                     'KPN918ec', 'KPN919ec', 'KPN921ec', 'KPN922ec',
                     'KPN923ec', 'KPN924ec', 'KPN925ec', 'KPN926ec',
                     'KPN927ec', 'KPN928ec', 'KPN929ec', 'KPN930ec',
                     'KPN931ec', 'KPN933ec', 'KPN934ec', 'KPN935ec',
                     'KPN939ec', 'KPN940ec', 'KPN941ec', 'KPN942ec',
                     'KPN943ec', 'KPN944ec', 'KPN945ec', 'KPN946ec',
                     'KPN947ec', 'KPN948ec', 'KPN950ec', 'KPN951ec',
                     'KPN952ec', 'KPN953ec', 'KPN954ec', 'KPN956ec',
                     'KPN957ec', 'KPN958ec', 'KPN959ec', 'KPN960ec',
                     'KPN962ec', 'KPN963ec', 'KPN965ec', 'KPN966ec',
                     'KPN967ec', 'KPN968ec', 'KPN973ec', 'KPN974ec',
                     'KPN975ec', 'KPN977ec', 'KPN978ec', 'KPN97ec-combo',
                     'KPN980ec', 'KPN981ec', 'KPN982ec', 'KPN983ec',
                     'KPN984ec', 'KPN986ec', 'KPN988ec', 'KPN989ec',
                     'KPN990ec', 'KPN994ec', 'KPN995ec', 'KPN997ec',
                     'KPN998ec', 'KPN999ec']

    self.isolate_to_row = dict((x,idx) for idx, x in enumerate(self.isolates))
    self.row_to_isolate = dict((v,k) for k,v in self.isolate_to_row.items())
    self.kmer_cnt = 94294557
    self.kmers = None
    print "loading kmer_to_col"
    self.kmer_to_col = self._load_kmer_to_col()
    print "loading col_to_kmer"
    self.col_to_kmer = dict((v,k) for k,v in self.kmer_to_col.items())
    self.M_dictionary = None
    self.M = self._load_M()
    self.M_small = self._load_M_small
    print "building labels"
    self.labels = self._build_lables()
    self.M_labels = np.array([self.labels[x] for x in self.antibiotics]).T
    self.index = self._build_index()
    self.masks = self._build_masks()
    print "5"
    self.results = {}

  def remove_duplicate_columns(M):
    a = M.T
    b = np.ascontiguousarray(a).view(np.dtype(
                          (np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return (a[idx]).T

  def _build_lables(self):
    #check if saved is here #very fast to build maybe leave as is
    ret = {}
    for idx, antibiotic in enumerate(self.antibiotics):
      ret[antibiotic] = np.zeros(len(self.isolates))
      logFile = self.logFileLoc + self.logFileNames[idx]
      with open(logFile, 'r') as f:
        log = [i.split() for i in list(f)]
        for l in log:
          if l[1]=='S':
            ret[antibiotic][self.isolate_to_row[l[0]]]=1
          else:
            ret[antibiotic][self.isolate_to_row[l[0]]]=2

      for idx, item in enumerate(ret[antibiotic]): #so wrong but whats right?
        if item == 0:
          ret[antibiotic][idx] = -1
        elif item == 1:
          ret[antibiotic][idx] = 0
        else:
          ret[antibiotic][idx] = 1
    return ret

  def _build_masks(self):
    ret = {}
    for antibiotic in self.labels:
      ret[antibiotic]= np.where(self.labels[antibiotic]>-1)[0]
    return ret

  def _build_index(self):
    return dict((antibiotic,np.where(self.labels[antibiotic]>-1)[0])
                  for antibiotic in self.antibiotics)

  def _load_kmer_to_col(self):
    if os.path.isfile(self.kmer_to_col_loc):
      return pickle.load(open(self.kmer_to_col_loc, 'rb'))
    else:
      print "error loading kmer_to_col"

  def _load_M_small(self):
     if os.path.isfile(self.M_smallLoc):
       return np.load(self.M_smallLoc)
     else:
       print "error loading M"

  def _load_M(self):
    if os.path.isfile(self.MLoc):
      return np.load(self.MLoc)
    else:
      print "error loading M"

  def build_col_to_kmer(self):
      self.col_to_kmer = dict((v,k) for k,v in self.kmer_to_col.items())

  def _build_M_dictionary(self):
    if os.path.isfile(self.indexDictionaryLoc):
      ret = pickle.load(self.indexDictionaryLoc)
    else:
      cnt = 0
      ret = {}
      for filename in os.listdir(self.kmerLoc):
        if '.sorted' in filename:
          print cnt
          cnt +=1
          with open(self.kmerLoc+filename) as reader:
            ret[filename.split('.')[0]] = dict(tuple(line.split())
                                               for line in reader)
      #with open('index.pickle', 'wb') as handle:
      #  pickle.dump(ret, handle)
    self.M_dictionary = ret

  def project_M(self,antibiotic):
    index = np.where(self.labels[antibiotic]>-1)
    return self.M[index,:]

def parallel_code(clf, M, labels, train, test):
  clf1 = clf()
  clf1.fit(M[train], labels[train])
  prediction = clf1.predict(M[test])
  return clf1, prediction, test

def run_in_parallel():
  skf = StratifiedKFold(labels, n_folds=10)
  parallel = Parallel(n_jobs=5, pre_dispatch='2*n_jobs')
  ret = parallel(delayed(parallel_code)(RandomForestClassifier,
                         M, labels, train, test) for train, test in skf)
  return ret

def make_correlation_matrix(kleb):
  itera = list(combinations_with_replacement(range(16),2))
  ret = {}
  for x,y in itera:
    if x!=y:
      ret[(x,y)]=[(row[x],row[y]) for row in kleb.M_labels if row[x]!=-1 and row[y]!=-1]
  lll = dict([(x,Counter(ret[x])) for x in ret.keys()])
  iii = [lll[x][(1,1)]/float(sum(lll[x].values())) for x in lll.keys()]
  jjj = [(x,lll[x][(1,1)]/float(sum(lll[x].values()))) for x in lll.keys()]
  output = np.zeros((16,16))
  for x in jjj:
    output[x[0][0],x[0][1]]=x[1]
    output[x[0][1],x[0][0]]=x[1]
  ax = sns.heatmap(output, cmap=plt.cm.Blues, linewidths=.1)
  ax.set_xticklabels(kleb.antibiotics, minor=False)
  ax.set_yticklabels(kleb.antibiotics, minor=False)
  fig = ax.get_figure()
  fig.show()
  return

if __name__ == '__main__':
    kleb = Klebsiella()

    clfs = []
    for antibiotic in kleb.antibiotics:
      print antibiotic
      labels = kleb.labels[antibiotic][np.where(kleb.labels[antibiotic]>-1)]
      clf = RandomForestClassifier()
      clf.fit(kleb.M[np.where(kleb.labels[antibiotic]>-1)],labels)
      clfs.append(clf)
