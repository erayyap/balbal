"""Proper nouns for the apostrophe rule of itn_text (balbal 0.6.8): Turkish given names and surnames (Wikidata, CC0), provinces and
districts, car brands and models, and company names. Only names that are rarely ordinary words, and whose inflected forms are not ordinary
words either, are listed. Separated by spaces, canonical spelling."""
PERSON = """
Aaron Abacıoğlu Abakay Abalı Abalıoğlu Abasıyanık Abazi Abbas Abbaslı Abbasoğlu Abdi Abdil Abdioğlu Abdul Abdulaziz Abdulhamid Abdulhamit
Abdulkadir Abdullah Abdurrahim Abdurrahman Abdurrezak Abdülhak Abdülhakim Abdülhamit Abdülkadir Abdülkerim Abdüssamed Abraham Acarbay
Acartürk Acemoğlu Acemyan Acer Acun Adanur Adanır Adatepe Adela Adem Ademoğlu Adiloğlu Adnan Adıcan Adıgüzel Afra Afyoncu Afşar Afşın Agha
Ahmad Ahmadi Ahmed Ahmet Ahrens Ahıskalı Ajda Akalay Akalın Akarcalı Akarcay Akat Akatlı Akay Akaydın Akbal Akbar Akbay Akbağ Akbaş
Akbaşoğulları Akbudak Akbulat Akbuğday Akbıyık Akcan Akcetin Akdağ Akdemir Akdeniz Akdoğan Akduman Akdülger Akel Akgöl Akgönül Akgöz Akgül
Akgün Akhun Akif Akile Akkal Akkale Akkartal Akkaya Akkaş Akkiraz Akkoyun Akkoç Akkurt Akkuzu Akkuş Akkılıç Akman Akmansoy Akmeşe Akpolat
Akpınar Aksa Aksel Akses Aksongur Aksoyoğlu Aksu Aksum Aksungur Aksöz Aksüt Aktan Aktas Aktay Aktaç Aktaş Aktekin Aktepe Aktuna Aktunç Aktuğ
Aktürk Aktürkoğlu Akyazı Akyol Akyürek Akyüz Akyıl Akçadağ Akçakoca Akçalı Akçaoğlu Akçatepe Akçay Akçiçek Akçura Akçıl Aküzüm Akınay Akıncı
Akışık Akşen Akşener Alaaddin Alaattin Aladağ Alagöz Alakuş Alankuş Alanson Alantar Alanyalı Alara Alasya Alataş Alatlı Alban Albayrak
Albayram Alben Alberto Aldemir Aldinç Aldo Aldoğan Aldıkaçtı Alec Alemdar Alemdaroğlu Alessandro Alex Alexander Aleyna Algan Algöz Algül Ali
Alican Alice Alicia Alihan Alin Alioğlu Alis Alisa Aliye Alizade Alişan Alkan Alkaş Alkin Alkor Alkım Allen Almina Almira Alnıaçık Alova
Alpacar Alpan Alparslan Alpaslan Alpay Alpaydın Alpdoğan Alper Alperen Alpsakarya Alptekin Alptuğ Altan Altay Altaylı Altaş Altunay Altunbaş
Altundal Altundağ Altunel Altuner Altunkaya Altunmeşe Altunok Altunoğlu Altuntaş Altuntop Altunyaldız Altuğ Altuğra Altıkat Altınbaş
Altınbilek Altındal Altınkaya Altınkaynak Altınköprü Altınok Altınoluk Altınordu Altınova Altınsoy Altıntaş Altıntop Altınışık Altıok
Altıparmak Alya Alyanak Alçelik Alço Amanda Amasyalı Anday Andaç Andican Andre Andrea Andrew Andy Andıç Ankaralı Anlağan Ann Anna Ansay
Ansel Antalyalı Antmen Anton Anıl Apaydın Arabacıoğlu Araksi Ararat Arca Arcayürek Arcıman Ardalı Ardıçlıoğlu Ardıçoğlu Arel Arf Ariadne
Arif Arife Arjen Arkaz Armando Arna Arolat Arpacı Arsebük Arseven Arslanoğlu Arslantaş Artemis Arthur Artin Artman Arto Artur Arvas Arya
Arıboğan Arıca Arıklı Arıkoğlu Arısoy Asaf Asarkaya Asena Ashok Asiltürk Aslanoğlu Aslantaş Aslantürk Aslıhan Asya Asyalı Asım Atabay Atabey
Atabeyoğlu Ataizi Atakan Atala Atalay Atalı Ataman Atamgüç Ataoğlu Ataselim Atatürk Atay Atayurt Ataç Ateşoğlu Atila Atilla Atiye Atlıhan
Attila Atuf Atun Atılgan Atınç Avcıoğlu Avi Avkıran Avni Avram Avşar Ayanoğlu Ayaşlı Aybaba Aybar Aybars Aybike Aybüke Aydar Aydemir Aydeniz
Aydoğdu Aydoğmuş Aydıner Aydınlı Aydınlıoğlu Aydınoğlu Aydınus Ayer Ayfer Aygan Aygül Aygün Aygüneş Ayhan Aykan Aykanat Aykut Ayman Aymaz
Aynacı Aynur Ayral Ayrancı Aysal Aysu Aysun Ayta Aytar Aytaç Aytaş Aytekin Ayten Aytunç Aytuğ Aytürk Ayvalı Ayvaz Ayvazoğlu Ayverdi Ayça
Ayçin Ayşe Ayşen Ayşenur Azak Azem Aziza Azizoğlu Azra Azrak Açıkalın Açıkel Ağaoğlu Ağbaba Ağgül Ağralı Ağırbaş Ağırman Aşcı Aşkar
Babaoğlan Babaoğlu Babür Baha Bahadır Bahadırlı Bahattin Bahire Bahri Bahçeci Bahçekapılı Bakırcıoğlu Bakırlıoğlu Balaban Balamir Balbay
Balcı Balcıoğlu Bali Balili Balin Balkan Balkar Balkaya Balkır Baloğlu Baltacı Baltazar Baluken Balıbek Bankoğlu Banu Barbara Barbaros
Barbarosoğlu Barbur Barkan Barlas Barnes Barokas Bartholomew Bartu Baruch Barutçiyan Barutçu Barçın Basilius Basmacı Basmacıoğlu Basri Bates
Batu Batuk Batum Batur Batıkan Bayar Bayazit Bayazıt Baybatur Baydar Baydemir Baydur Baykan Baykara Baykurt Bayman Baynazoğlu Bayraktar
Bayraktaroğlu Bayraktutan Bayrakçı Bayramlı Bayramoğlu Baytok Baytop Bayur Bayülgen Baçova Bağcan Bağcı Bağdat Bağdatlı Bağrıaçık Başara
Başaran Başbuğ Başer Başevirgen Başeğmez Başkaya Başol Başoğlu Başpınar Başterzi Baştopçu Baştuğ Baştürk Başyiğit Becky Bedia Bedir
Bedrettin Bedri Bedriye Bedük Begüm Behar Behice Behiç Behram Behramoğlu Behzat Bekaroğlu Beken Beker Bekir Bekiroğlu Bektaş Belek Belen
Belger Belgin Belgün Belkin Belkıs Belçim Belözoğlu Benay Bengisu Bengü Benhabib Benjamin Benyamin Berat Berdibek Berfin Berge Berger
Bergman Bergüzar Beria Beritan Berivan Berk Berkan Berkant Berkay Berkin Berkok Berktaş Berkun Berre Berrin Bertuğ Berç Besime Betty Betül
Beyazoğlu Beyaztaş Beyazıt Beyhan Beytullah Beyza Beğik Beşerler Beşikci Beşikçi Beşirli Beşiroğlu Beşok Beştaş Bianca Bihter Bilal Bilican
Billy Bilsel Binali Bingül Binnaz Birant Birben Birgül Birhan Birkan Birol Birsel Birsen Bogdan Bolat Bolek Boluğur Bonomo Boral Boralıoğlu
Boran Boratav Boris Borova Bosse Bostancıoğlu Botan Boyacıoğlu Boyle Boyraz Bozarslan Bozatlı Bozbay Bozbey Bozdağ Bozdemir Bozdoğan Bozer
Bozgeyik Bozkurt Bozok Bozoğlan Bozoğlu Boztepe Bozyel Bozyiğit Boğaç Bradley Brandt Bulakbaşı Bulaç Buldan Bulutoğlu Bulutoğluları Bumin
Burgaz Burhan Burhaneddin Burhanettin Burian Burnham Burçak Burçin Buse Buğra Buğrahan Byron Béla Böke Bölek Bölükbaşı Börekçi Bülend Bülent
Bünyamin Bürsin Büyükak Büyükataman Büyükbayrak Büyükbayraktar Büyükberber Büyükburç Büyükciğer Büyükkaya Büyüktortop Büyüküstün Büşra
Büşranur Bıyıklıoğlu Bıçakçı Caba Cafer Cahid Cahide Cahit Camcı Campbell Canan Canbaz Canbaş Canbey Canbulat Candar Candaş Candide Canel
Caneroğlu Canikli Cankat Canko Cankurt Canpolat Cansel Cansever Cansu Cantürk Carmen Caroline Casalini Cavadzade Cebeci Cebrail Cehan Celal
Celalettin Celep Cemal Cemalettin Cemil Cemile Cemşir Cenap Cengiz Cenk Ceran Cerit Ceritoğlu Cerrahoğlu Cevat Cevdet Cevizci Ceyhan Ceyhun
Ceylin Chiara Chloe Christian Christine Cihangir Cihannur Cindoruk Civan Civelek Cohen Coker Colin Cora Coryell Coşkun Coşkuner Cristiano
Cumali Cumalı Cumbul Cumhur Cüceloğlu Cüneyt Daimagüler Dalaklı Dalkılıç Dalli Daloğlu Dalrymple Damir Danabay Daniel Danişmend Danyal Daren
Dario Darius Darko Davarcı Dave Daver Davud Davut Davutoğlu Dawkins Dayıoğlu Dağdaş Dağdelen Dağdeviren Dağhan Dağlaroğlu Dağtekin Daştan
Dedeoglu Dedeoğlu Delibaş Deligöz Deliktaş Deliorman Demiral Demiralay Demiray Demirağ Demirbakan Demirbay Demirbağ Demircan Demircioğlu
Demirdağ Demirel Demirer Demirezen Demirezer Demirhan Demirkaya Demirkol Demirkubuz Demirkıran Demirok Demiroğlu Demirsoy Demiröz Denis
Denizhan Denker Dennis Dereli Derelioğlu Derici Dermancıoğlu Dervişoğlu Destici Develi Develioğlu Değirmencioğlu Diamond Diana Dicle Didem
Dikeç Dilan Dilipak Dilmen Dilruba Dilşa Dimitriadis Dino Dinçel Dinçer Dirik Diyarbakırlı Dizdaroğlu Dolen Domaç Domenico Dora Dormen
Dorukhan Doğacan Doğanay Doğandemir Doğaner Doğançay Doğramacı Doğruer Doğrusöz Doğruyol Dudley Dudu Dumanoğlu Dumbe Dunford Dunne Durakoğlu
Dursunoğlu Durukan Dutlulu Dökmeci Döşkaya Dülgeroğlu Dümer Düvenci Düzgören Düzyatan Eaton Ebubekir Ebüzziya Ecem Ecevit Eczacıbaşı Edanur
Edgar Edgü Edgüer Edis Ediz Edman Edward Efe Efecan Efeoğlu Efes Efgani Efkan Egbert Ege Egeli Ehlers Ekici Ekim Ekizoğlu Ekmekçioğlu Ekrem
Ekren Ekşioğlu Elanur Elgün Elia Elias Eliaçık Elik Elis Elisa Elissa Elmacı Elmaz Elvin Elvira Emanuel Emel Emeç Emina Emine Emire Emirhan
Emiroğlu Emma Emmanuel Emrah Emrullah Enes Enez Enginyurt Engür Enver Enç Eracar Eralp Erarslan Eraslan Eray Erbakan Erbay Erbaş Erberk
Erbil Ercan Erce Erciyes Ercüment Erdal Erdan Erdemir Erdemoğlu Erdinç Erdoğan Erdoğmuş Erel Erencan Erendiz Erener Erenli Erenoğlu Erensoy
Erensü Erez Ergene Ergenekon Ergeneman Ergenç Ergezen Erginci Ergun Ergü Ergüder Ergül Ergülen Ergün Ergüney Ergüven Ergüçlü Erhan Eriksen
Eriç Erişçi Erkakan Erkal Erkan Erkanal Erkin Erkmen Erkoç Erksan Erkurt Erkut Erman Ernestine Ernst Eron Eroğlu Ersan Ersay Ersin Ersoy
Ersu Ersun Ersöz Ertan Ertegün Ertekin Ertem Erten Ertunç Ertuğ Ertuğruloğlu Ertöz Ertürk Erva Eryiğit Eryurt Eryüksel Eryılmaz Erzen Erçe
Erçel Erçetin Erçin Erşahin Erşan Esad Esat Escobar Esenboğa Esendemir Esentürk Esfender Esila Esma Esmeray Ethem Eva Evan Evans Evcen
Everest Evgin Evlioğlu Evliyaoğlu Eyice Eylül Eymen Eyüboğlu Eyüpoğlu Ezra Eğilmez Eğribayat Eğridere Eşfak Fadime Fahir Fahrettin Fahriye
Faik Faner Farah Farhad Faruk Farzan Fatima Fatime Fatin Fatma Fatmagül Faysal Fazlıoğlu Fedon Fehime Fehmi Fellahoğlu Ferat Ferdi Ferham
Feridun Feriha Ferit Fernando Fersoy Fethullah Fevzi Fevziye Feyzullah Figen Fikret Filiz Finkel Firdevs Fischer Fitnat Flynn Fosforoğlu
François Freddy Fuad Fuat Fuchs Fulya Funda Furkan Fındıkoğlu Fırat Gabriel Gaffar Gafur Gamze Garan Garbis Gazanfer Gazi Gebeş Gencebay
Gencer Genco Gençer Gençoğlu Gençtürk George Georges Georgios Geredeli Gerger Gergerlioğlu German Germaner Germen Germiyanoğlu Gerçeker
Geçkil Giannis Gilbert Giovanni Giray Girgin Giritlioğlu Goncagül Goran Gregory Greta Grigoriadis Gruda Guido Guliyev Guran Gustaf Gustav
Gökalp Gökay Gökbakar Gökberk Gökbulut Gökcan Gökce Gökdeniz Göker Gökgöz Gökhan Gökmen Göknel Göknil Göksun Göksungur Göktan Göktepe
Göktuna Göktürk Gökyay Gökçe Gökçek Gökçen Gökçer Gökçin Gökçınar Gömleksizoğlu Göncü Gönen Gönenç Göral Gözen Gözütok Göğem Göğüş Gübür
Gücer Güdül Gülay Gülbahar Gülben Gülberk Gülcan Gülcü Güldemir Gülderen Gülek Gülenay Gülender Gülenç Güleroğlu Güleryüz Gülfem Gülgen
Gülhan Gülhayat Güliz Gülkız Güllü Güllüce Gülnaz Gülpınar Gülru Gülsel Gülselam Gülsen Gülser Gülseren Gülsin Gülsoy Gültan Gültekin Gülten
Gültepe Gülyanar Gülçin Gülşah Gülşe Gülşen Gümeli Gümrü Gümüşay Gümüşdal Gümüşdağ Gümüşdere Gümüşer Günal Günaltay Günay Günday Günder
Gündoğan Gündoğar Gündoğdu Günel Güner Güneri Güneyligil Güneyman Güneysu Güngör Günseli Günsu Güntekin Günter Gürak Güral Güran Güray
Gürayman Gürbey Gürbüz Gürbüzer Gürcan Gürdal Güreli Gürer Güres Gürhan Gürkan Gürkaynak Gürman Gürmen Gürol Gürpınar Gürsel Gürses Gürsey
Gürsoy Gürsu Gürtuna Gürzap Gürçağ Gürçınar Gürün Güveli Güvener Güvenç Güzelses Güzin Güçbilmez Gıyasettin Habib Habil Hablemitoğlu Hacer
Hacıeyüpoğlu Hacımüftüoğlu Hadiye Hadımoğlu Hafize Hafsa Hagi Halide Halil Halima Halman Haluk Halûk Halıcı Hamdi Hamdullah Hamid Hamide
Hamit Hamza Handan Hande Hans Hansen Harun Harvey Hasan Hasdemir Hasgüler Hasipoğlu Haspolat Hassan Hastürk Hasçelik Hasırcı Hatice
Hatımoğulları Havutçu Havva Haydar Hayrettin Hayri Hayrullah Haywood Hazal Hazar Hector Hein Hekimoğlu Helga Helmi Helmut Helvacı
Helvacıoğlu Henri Henry Heper Hepileri Hepşen Hermann Hicran Hilda Hillary Hilmi Hindiyerli Hira Hiram Hirsch Hocaoğlu Horasan Hosseini
Hotar Hozan Huduti Hughes Hulûsi Hun Huriye Hurşit Husret Hussein Hüdaverdi Hülagü Hülya Hümeyra Hünel Hürrem Hüsamettin Hüseyin Hüseyinoğlu
Hüsnü Hüsrev Hıdır Hıfzı Hıncal Hızır Ibis Ibrahim Ida Idris Ikeda Ildır Ilgın Ilıcak Ilıcalı Ingham Ioannis Irak Irene Iris Isaac Isabella
Ishak Islam Ismail Ivan Ivana Işılay Işınsu Jack Jackson Jacob Jacqueline Jacques Jakob Jale Jamie Jan Janet Janko Jason Jean Jehan Jem
Jerome Jessica Joan Johannes Johansen John Jonny Josef Joseph Julian Julide Julie Julio Jülide Kaan Kabakçı Kabakçıoğlu Kabasakal Kaboğlu
Kacin Kadiroğlu Kadri Kadıoğlu Kaftancıoğlu Kahyaoğlu Kais Kalafatoğlu Kalaycıoğlu Kalaç Kalecik Kalelioğlu Kalkancı Kalkavan Kaloğlu
Kalyoncu Kamal Kaman Kamiloğlu Kandemir Kaner Kansu Kantarcı Kantemir Kanık Kapu Karaali Karaarslan Karaaslan Karaavcı Karaağaç Karaağaçlı
Karabacak Karabay Karabaş Karabekir Karabulut Karaböcek Karabıyık Karacabey Karacaoğlu Karadayı Karadağ Karadağlı Karademir Karadeniz
Karadere Karaderi Karadoğan Karaduman Karaer Karagülle Karahan Karahancı Karahasan Karaibrahimgil Karaismailoğlu Karakaş Karakaşlı Karakoyun
Karakoyunlu Karakullukçu Karakurt Karakuş Karaköse Karaloğlu Karamete Karamustafa Karaosman Karaosmanoğlu Karaoğlan Karaoğlanoğlu Karaoğlu
Karapınar Karasu Karasungur Karataş Karataşlı Karatekin Karatepe Karavelioğulları Karayalçın Karayel Karayılan Karayılmaz Karaçayır Karcı
Karlıdağ Karmen Karni Karslı Karslıoğlu Karışlı Kasapoğlu Kaslowski Kaspar Kasım Kathleen Katipoğlu Kavaklı Kavala Kavlak Kavuncu Kawa
Kayaalp Kayabaşı Kayacan Kayacık Kayadelen Kayalı Kayaman Kayaoğlu Kayapınar Kayasu Kaygusuz Kayhan Kaymakçı Kaypakkaya Kayra Kayserili
Kayıhan Kayışoğlu Kazancıgil Kazım Kağan Kağıtçıbaşı Kaşıkçı Kegan Kekilli Keles Kelleci Kemal Kemalettin Kemaloğlu Kenan Kendirli Kenneth
Kenter Kentmen Kepenek Kerestecioğlu Kerim Kerimoğlu Kesal Kesimal Keskiner Ketenci Kevin Kezban Kezer Keziban Keçeci Keşir Keşmir
Khachaturian Khalidi Khalil Khan Khosravi Kiper Kirazcı Kirazlı Kirazoğlu Kiremitçi Kiyan Kobal Kobaş Kocabaşoğlu Kocabıyık Kocacık Kocaer
Kocagöz Kocak Kocaoğlu Kocatepe Kocatürk Kocuk Kodal Kohen Kolat Kolcuoğlu Koldaş Kolivar Koloğlu Kolukısa Konal Koncuk Konuralp Konyalı
Koral Koraltürk Koray Korkut Korutürk Kosal Kosova Kosovalı Kosta Kostas Kotil Kozlu Kozluca Koçak Koçal Koçbay Koçdemir Koçibey Koço
Koçoğlu Krause Kris Kristen Kuban Kubat Kubilay Kulaksızoğlu Kulin Kunt Kunter Kurdaş Kurdoğlu Kurtaş Kuruoğlu Kuruçay Kutalmış Kutay
Kutluay Kutlubey Kutluer Kutluğ Kuşoğlu Kuşçu Kuşçubaşı Kâmil Kâzım Kökdemir Köklükaya Köksalan Köprülü Körmükçü Köroğlu Köstepen Köybaşı
Köymen Kübra Küpeli Küpçü Kürten Kürşad Kürşat Kütahyalı Küçükakyüz Küçükel Küçükkaya Küçükoğlu Küçükçağlayan Küçükşahin Kılınç Kılıçarslan
Kılıçaslan Kılıçdaroğlu Kılıçsallayan Kınacı Kıpçak Kırantepe Kıraç Kırağası Kırbaş Kırca Kırcalı Kırdar Kırkpınar Kırmızıtaş Kırımlı
Kırımlıoğlu Kırşan Kısakürek Kıvanç Kıvrıkoğlu Kızkapan Kızıldağ Kızıldemir Kızılhan Kızılkaya Kızılkılıç Kızılok Kızıloğlu Kızıltan
Kızıltaş Kızıltuğ Kızılyürek Kızılırmak Kışlalı Lahn Lamar Latifi Laura Laçiner Leblebici Lefter Leman Lemi Lena Leonidas Leopold Lerzan
Leslie Levendoğlu Levi Levin Leyla Lici Lisa Liza Lokman Lucie Lucien Lukas Lynne Léon Lüleci Lütfi Lütfiye Lütfullah Lütfü Maan Maciej
Macit Maga Mahinur Mahiye Mahmoud Mahmud Mahpeyker Majid Makbule Malachi Malay Malbora Malkoç Mamat Manfred Mansur Mansız Marc Marcel Marek
Margulies Mari Mariana Mariya Marmara Marquardt Marsel Martino Marvin Maryam Mato Mattei Mauro Max Mayda Maçin Mağden Mebrure Mecdi Mecit
Mediha Medina Medine Mehdi Mehman Mehmed Mehmet Mekki Melahat Melen Melia Meliha Melike Melina Melisa Melissa Meliz Melo Meltem Memduh
Memili Memiş Memişoğlu Menemencioğlu Mengi Mengü Mengüç Mensur Mente Meral Merdan Mergen Meriç Mermut Merter Mertoğlu Mervan Meryem
Mesutoğlu Mete Metehan Midhat Mihal Mihriban Mikail Mikdad Miray Miraç Mirko Miro Miroğlu Mithat Mocan Mohamed Mohammad Mollaoğlu Mona
Monika Monique Moradi Moralı Moralıoğlu Moray Morgül Moritz Morkaya Morkoç Moshe Muammer Muhamed Muhammed Muhammet Muharrem Muhittin Muhsin
Muhyiddin Mumay Mumcu Mungan Murat Muratoğlu Muriqi Mursal Mursaloğlu Murtezaoğlu Musa Musaoğlu Mustafa Mutluer Müderrisoğlu Müfit Müftüoğlu
Müge Müjdat Müldür Mümtaz Münir Münire Mürsel Mürvet Müslüm Mıhçıoğlu Nacar Naci Naciye Nader Nadim Nadine Nafiz Nagehan Nahum Naile Naim
Nakiye Nalan Nalbantoğlu Namoğlu Namık Nana Nasser Nasuhi Nasuhoğlu Naum Nayır Nazan Nazi Nazif Nazlıaka Nazmi Nazmiye Nazım Naşide Naşit
Nebil Necat Necdet Necef Necil Necip Necla Necmettin Necmi Nedim Nedime Nedret Nejat Nergiz Neriman Nermin Nesibe Nesip Neslihan Neslişah
Nesrin Neubauer Nevra Nevzat Nezihe Neşecan Neşet Nicolas Niels Nigar Nihal Nihat Nikola Nilgün Nina Nisan Nisanur Niyazi Nizamettin Nizar
Nobre Nova Noyan Nubar Nuh Nuhoğlu Nukan Numan Numanoğlu Nur Nurcan Nurdoğan Nurettin Nurgül Nurhan Nurhayat Nuri Nurkan Nursel Nurten
Nurullah Nusret Nüket Nükhet Nüzhet Oben Ocak Ocan Odabaş Odabaşı Odman Ofelya Oflaz Ogeday Ogün Ohannes Oker Oktar Oktay Okyar Okyay Olcan
Olcay Olcayto Olgaç Olivia Oluç Omar Onat Ongu Onuralp Oraloğlu Orcan Orel Orhan Orhun Ortakaya Ortaylı Ortaç Ortaçgil Oruçoğlu Oskar Osman
Osmanoğlu Oytun Oğan Oğulcan Oğuz Oğuzcan Oğuzhan Pakdemirli Pakdil Pakize Palabıyık Pamukçu Paris Pasha Patrick Patterson Paul Pavey Pavlos
Paçacı Paşalı Pegg Pehlivanlı Pehlivanoğlu Pekcan Pekdemir Pekel Peker Pekin Pekkan Pektaş Pektemek Pektürk Pekuysal Pekşen Penbe Pepe
Perihan Perinçek Pertev Pervin Petar Petekkaya Peter Peyami Philipp Philippe Pierre Pirhasan Polat Polatkan Poyrazoğlu Preston Pringle Pulak
Pınar Pınarbaşı Rabia Rachel Ragıp Rahman Rahmi Rahşan Raif Rama Raman Ramazan Ramazanoğlu Ramiz Ramoğlu Rana Rankin Rashid Rasim Rasin Rauf
Ravil Ravza Rayyan Raşit Recai Recep Reddy Refii Refik Refika Reha Reinhart Remzi Remziye Renan Renda Resul Reza Reçber Reşat Reşid Richard
Richardson Rickie Rifat Roberta Rodrik Rogers Rojda Rosie Ross Roy Ruben Ruf Ruhsar Ruken Rukiye Rutkay Ruşen Rümeysa Rüstem Rüşdü Rüştü
Rıdvan Rıfat Sabahat Sabahattin Sabiha Sabri Sabuncu Sabuncuoğlu Sadi Sadioğlu Sadir Sadrettin Sadri Saffet Safiye Sagar Sahipoğlu Saide
Saim Sakallıoğlu Sakaoğlu Sakman Sakık Sakıp Salah Saleh Salem Salgut Saliha Salihoğlu Salman Salmanlı Salter Saltoğlu Saltuk Salıcı Sam
Samancı Samast Samat Samed Samet Sami Samia Samir Samiye Samlı Samsa Samsunlu Samuel Sancaklı Sancaktutan Sancar Sander Sangu Sanlı
Saracoğlu Sarah Saraç Saraçoğlu Sargın Sarhan Saruhan Sarıalioğlu Sarıbaş Sarıca Sarıeroğlu Sarıgül Sarıkaya Sarıoğlu Sarısözen Sarıtaş
Sarıtekin Sarıyıldız Savranbaşı Sayari Sayek Saygun Saygıner Saylan Sayıner Sayışman Sazak Sağman Sağtürk Sağun Sağyaşar Sağınç Sağıroğlu
Scheinhardt Schmidt Schmitt Schmitz Scottie Sebahattin Sebük Sebüktekin Seferinoğlu Seki Selahaddin Selahattin Selami Selcan Selda Selek
Selena Selim Selimoğlu Selina Selman Selmani Selmin Selva Selışık Semahat Semerci Semercioğlu Semih Semiha Semiramis Semra Senai Sencer
Sercan Serdar Serdaroğlu Serenay Serengil Seres Sergen Sergius Serhan Serhat Sermin Serpil Sertab Sertan Sertaç Sertel Serter Sertçelik
Sesigüzel Setenay Seth Seval Sevcan Sevik Seydaoğlu Seyfettin Seyfi Seyfullah Sezai Sezar Sezen Sezgin Shahin Shane Sibel Sidney Silan
Silvan Simavi Simon Sina Sinan Sinanoğlu Sinem Sipahioğlu Sivar Siyavuş Sofiya Sokullu Somer Sonay Soner Songül Sonia Sonja Sonya Sophie
Soriano Soysal Soğancıoğlu Spain Spencer Stella Stephan Stephen Steven Stevenson Suat Suavi Subhi Sucuoğlu Sudi Suleiman Sultan Suna Sunal
Sunay Sungar Sunter Suver Suzy Sökmen Sökmensüer Sönmez Söylemezoğlu Söyüncü Sözen Sözer Sözeri Söğütlügil Süalp Süer Süheyl Süheyla
Süleyman Süme Sümer Sümeyye Süner Süreya Süreyya Sürmeneli Sütçüoğlu Sıddık Sıdıka Sıla Sıraç Sırdar Sıtkı Tabanlıoğlu Tacan Taha Tahaoğlu
Tahir Tahsin Talaslıoğlu Talat Talha Talu Tamar Tamara Tanay Tandoğan Taner Tangöze Tanju Tanrıbilir Tanrıkulu Tanrıseven Tanrısevsin
Tanrıyar Tanrıöver Tanrıöğer Tansel Tansu Tanya Tanyel Tanyeli Tanyeri Tanyolaç Tanyıldızı Tanzer Tarakçı Tarancı Taranç Targan Tarkan
Tarlabaşı Tarık Tatar Tataroğlu Tatlıoğlu Tatlıses Tatlıtuğ Tavgaç Taviloğlu Tavukçuoğlu Tayfun Tayfur Taylan Tayyar Tayyip Taçkın Taşbaş
Taşdelen Taşdoğan Taşdöğen Taşkan Taşkent Taşkesenlioğlu Taşova Taşoğlu Taşpınar Taşyürek Taşçıer Teber Tekdal Tekelioğlu Tekinoktay Tekçe
Temelkuran Temizel Temizkan Temür Teoman Tercan Terzioğlu Tevfik Teymur Tezcan Tezer Teziç Theodor Thomas Tiago Tibet Tigin Tilbe Tilmaç
Timo Timtik Timur Timuroğlu Timurçin Timuçin Tiritoğlu Tito Togay Toker Tokgöz Toksöz Toköz Tolon Tomislav Tomris Tonga Tonguç Topaktaş
Topaloğlu Topel Topkaya Toplusoy Toptaş Topçuoğlu Toros Tosun Tosuner Toyran Tozkoparan Troy Tuan Tuana Tugay Tuma Tunaboylu Tunalı Tunca
Tuncay Tuncer Tunga Tunç Tunçalp Tunçay Tunçboyacıyan Turacı Turak Turan Turani Ture Turgay Turgut Turhan Turner Tutumluer Tuğba Tuğberk
Tuğcu Tuğkan Tuğrul Tuğutlu Tuğçe Törel Törer Tüfekçi Tüfekçioğlu Tükel Tülay Tülek Tülin Tümay Tümer Tümtürk Türeci Türel Türemen Türk
Türkali Türkan Türkaslan Türkay Türkcan Türkdeniz Türkeli Türker Türkeri Türkeş Türkkan Türkmen Türkoğlu Türkpençe Türksoy Türkçüoğlu Türköz
Tütüncü Tüzemen Tınaz Ubeydullah Uday Ulaç Ulubay Uludaş Ulukaya Ulunay Ulusan Ulutaş Uluç Uluçay Uluğ Uncuoğlu Ungan Ural Uraz Urgancıoğlu
Urhan Usar Usluer Usman Ustaoğlu Utkan Utku Utkun Uygül Uzel Uzerli Uzunalioğlu Uzunhan Uzunçarşılı Uğur Uğuralp Uğurcan Uğurtan Uşaklıgil
Vahap Vahdettin Vahid Vahide Vahit Vanlı Vano Varank Vardar Vardarlı Vargas Vargın Varley Varlı Varol Vasilis Vedat Vehbi Velioğlu Ventura
Verşan Veysel Veysi Victoria Vildan Villeneuve Vitali Vogt Václav Vüqar Vüsat Weber Winslow Wright Xuan Yahya Yakup Yakupoğlu Yalvaç Yalçın
Yalçındağ Yalçıner Yalçınkaya Yalçıntaş Yamantürk Yanaç Yanik Yanya Yapıcıoğlu Yarman Yarıcı Yasin Yasir Yasmin Yasmina Yatağan Yavaşoğlu
Yavaşoğulları Yavuz Yaycı Yazgan Yazıcıoğlu Yağtu Yaşargil Yaşaroğlu Yedikardeş Yekaterina Yekta Yelda Yeldan Yeliz Yeltekin Yenal Yenel
Yenice Yenigün Yeninci Yeniçay Yerlitaş Yeğin Yeşilbaş Yeşilmen Yeşiltaş Yeşilyurt Yeşilçay Yeşilöz Yiğitcan Yiğithan Yoloğlu Yorgancıoğlu
Yorgos Youssef Yoğurtçu Yula Yunus Yurdaer Yurdakul Yurtalan Yurtdaş Yurtseven Yusuf Yusufi Yusufoğlu Yuvakuran Yönder Yörükoğlu Yücedağ
Yüceer Yücel Yücelen Yücesan Yücesoy Yücetan Yücetürk Yüksel Yükseloğlu Yülek Yüsra Yüzbaşıoğlu Yılancı Yıldıran Yıldızhan Yılmaz Yılmazer
Yırcalı Zabunyan Zafir Zahide Zaim Zakaryan Zaki Zarife Zaza Zehra Zeid Zekeriya Zelal Zeliha Zeliş Zerrin Zeybekoğlu Zeydan Zeyid Zeynel
Zeyrek Zeytinoğlu Zincidi Ziya Zoran Zuckerman Zuhal Zübeyde Zühal Züleyha Zülfiye Zülfü ismet Çadırcıoğlu Çaka Çakmakoğlu Çakmakçı Çakır
Çakıroğlu Çalhanoğlu Çallı Çalık Çalıkuşu Çalımbay Çalışlar Çambel Çamcı Çamdalı Çamkerten Çamlıbel Çanakçı Çandar Çanga Çapan Çapanoğlu
Çapar Çapoğlu Çarmıklı Çatak Çavaş Çayan Çayoğlu Çayırlı Çağan Çağaptay Çağatay Çağla Çağlayan Çebi Çelebi Çelebioğlu Çelikay Çelikbilek
Çelikdemir Çelikel Çeliker Çelikkan Çelikkaya Çelikkol Çepni Çerkes Çerkezoğlu Çetin Çetinbaş Çetindağ Çetinel Çetiner Çetinkaya Çetintaş
Çevikel Çeviköz Çeçen Çeşmelioğlu Çeşmioğlu Çiftpınar Çiftçioğlu Çilingiroğlu Çiloğlu Çisil Çitil Çizmeci Çiçekoğlu Çiğdem Çobanoğlu Çoker
Çolakoğlu Çolpan Çongar Çorlulu Çoruh Çorumlu Çuhadaroğlu Çulha Çulhaoğlu Çölgeçen Çöpdemir Çıracı Çıtak Çıvgın Öcal Öcalan Ökkeş Ökmen
Öktem Ökten Ömer Ömeroğlu Ömürcan Önal Öndeş Önen Öney Öngün Öniz Öntürk Önver Ötenel Özacar Özak Özal Özalan Özalp Özarslan Özarı Özaslan
Özat Özata Özatay Özateş Özatlı Özay Özaydınlı Özbahar Özbakır Özbaş Özbek Özben Özberk Özbir Özbiçer Özbudak Özcan Özdal Özdamar Özdağ
Özdağlar Özdem Özdemir Özdemiroğlu Özdil Özdilek Özdoğan Özelçi Özener Özenç Özer Özeren Özerman Özgan Özge Özgecan Özgen Özgener Özgentürk
Özgenç Özgökçe Özgüden Özgündüz Özgüneş Özgürgün Özgüç Özhan Özhaseki Özil Özince Özipek Özkal Özkan Özkara Özkasnak Özkaya Özkazanç Özker
Özkurt Özkök Özlen Özmert Özoğlu Özoğul Özoğuz Özpetek Özpirinçci Özpolat Özpınar Özseven Özsoy Öztan Öztaş Öztaşdelen Öztekin Öztepe
Öztorun Öztrak Öztuna Öztunç Özturan Öztürk Öztürkmen Özvar Özveren Özyakup Özyavuz Özyazıcı Özyağcılar Özyaşar Özyiğit Özyurt Özyüksel
Özyıldırım Özyıldız Özyılmazel Özçimen Özçivit Özüdoğru Özışık Özşahin Özşener Öçal Öğütçü Ülgen Ülger Ülgün Ülker Ümmiye Ümmühan Ümran
Ünalmış Ünder Ünel Üner Ünlüer Ünlüoğlu Ünsal Ünsalan Ünüvar Üregül Ürgüplü Ürkmezgil Üründül Üstünel Üstüner Üstünkaya Üstüntaş Üçok
İbrahim İbrahimli İclal İdris İkizer İkizkaya İlbay İlbaş İldem İlhami İlhan İlksen İlyas İmamoğlu İmran İnalcık İnanoğlu İnceman İnceoğlu
İncesu İnceöz İncinur İnkaya İnönü İpekçioğlu İplikçi İrem İrtegün İshakoğlu İskender İskenderoğlu İslam İslamoğlu İslimyeli İsmailoğlu İyem
İynemli İzmirli İzzettin İçöz İğrek İşcan İşeri İşikırık İşkan İşsever Şaban Şadi Şahali Şahan Şahin Şahinbaş Şahingöz Şahinkaya Şahinoğlu
Şahverdi Şakar Şakir Şamdereli Şamil Şamlıoğlu Şanal Şanlıtürk Şarık Şatıroğlu Şebnem Şefik Şefika Şehnaz Şehrazat Şehsuvar Şekercioğlu
Şekeroğlu Şekibe Şekip Şemsettin Şenalp Şenay Şencan Şendil Şenel Şener Şeneş Şengel Şengil Şengör Şengöz Şengül Şengün Şenlendirici Şennur
Şenocak Şenol Şenova Şenses Şentop Şentürk Şeras Şerdil Şeren Şerife Şermin Şevket Şevval Şeyda Şeyhmus Şimşek Şinasi Şule Şükriye Şükrü
Şıkel
""".split()
OTHER = """
Adana Adapazarı Afyonkarahisar Aksaray Akyurt Aliağa Alsancak Altunizade Altındağ Amasya Anadolu Ankara Antalya Antep Ardahan Arnavutköy
Artvin Arçelik Austral Axa Ayaş BMW Bahçelievler Balçova Balıkesir Bartın Batıkent Bayburt Bayon Bayrampaşa Bayındır Başakşehir Beko Bergama
Beykoz Beylikdüzü Beyoğlu Beypazarı Beşiktaş Bilecik Bingöl Bitlis Bolu Bornova Bostanlı Boyner Buca Burdur Büyükçekmece Captur Ceed Chery
Citroen Cooper Corolla Corsa Cupra Dacia Darıca Defacto Didim Diyarbakır Doblo Duster Düzce Egea Elazığ Elmadağ Enyaq Erenköy Eryaman
Erzincan Eskişehir Etimesgut Eyüpsultan Facebook Fethiye Fibabanka Fiorino Florya Ford Foça Gaziantep Gaziosmanpaşa Gebze Gemlik Giresun
Girne Google Gümüşhane Güngören Güzelbahçe HSBC Hakkari Haymana Honda Huawei Hyundai ING Ibiza Insignia Instagram Iğdır Kadıköy
Kahramankazan Kahramanmaraş Karabağlar Karaburun Karabük Karaman Karatay Kavacık Kayseri Kemalpaşa Keçiören Kia Kocaeli Konya Kozyatağı Kuga
Kurtköy Kuşadası Kütahya Küçükçekmece Kınık Kırıkkale Kırşehir Kızılay Kızılcahamam Lefkoşa Leon Lexus Lüleburgaz MNG Malatya Mamak Manavgat
Maraş Mardin Marmaris Mazda Mecidiyeköy MediaMarkt Megane Mercedes Microsoft Mitsubishi Mokka Multivan Muğla Nallıhan Narlıdere Netflix
Nevşehir Nissan Niğde Nişantaşı Octavia Opel Ortaköy Osmaniye PTT Peugeot Polatlı Porsche QNB Qashqai Renault Rize Romeo Rover Samsun
Samsung Sandero Sarıyer Scala Scirocco Seferihisar Shopify Siirt Sincan Spotify Suadiye Sultanbeyli Sultangazi Suzuki Talas Taliant Tarsus
Tekirdağ Telekom Tesla Tiggo Togg Torbalı Toyota Tunceli Türksat Urfa Urla Vitara Vodafone Volvo Waikiki WhatsApp Xiaomi Yalova Yemeksepeti
Yenimahalle YouTube Yozgat Zeytinburnu Zonguldak Çamlıca Çanakkale Çatalca Çayyolu Çiğli Ümitköy Üsküdar İnegöl İskenderun İzmit Şahinbey
Şehitkamil Şişli Şırnak
""".split()
# names that look like <name><suffix> but are names themselves (lowercase): never split
NEVER = """
abdulla ademi atakanı nurdan selimi taneri ıbrahimi
""".split()
