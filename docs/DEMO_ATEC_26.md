# Artemis-X — démo ATEC’26 : lancement, tournage et pitch

Analyse du 11 octobre 2026, à partir du **Atec'26 Specification Book.pdf** fourni (11 pages), du code actuel et des rapports R4–R6. Les exigences ci-dessous viennent du PDF ; les durées de tournage et le choix des plans sont nos recommandations. Cette préparation ne lance pas une mission et ne constitue pas une nouvelle validation end-to-end.

## 1. Ce que le jury demande réellement

- **Pages 5–7 :** percevoir localement, projeter les détections dans un repère spatial, transmettre des coordonnées compactes sur un réseau limité avec pertes, afficher des données reçues avec leur âge ou leur incertitude.
- **Page 6 :** prototype entièrement logiciel, chaîne perception → télémétrie → dashboard, contraintes fortes de bande passante. Modèle, langage, protocole et visualisation restent libres.
- **Page 7 :** aucune API cloud externe pour la perception ; sockets/scripts réseau implémentés par l’équipe.
- **Page 9 :** dépôt de code **et sorties des tests automatisés** pour la présélection ; 7 finalistes ; **7 minutes de présentation technique en direct puis 3 minutes de Q&A**. La chaîne complète doit fonctionner en live.
- **Page 10 :** cinq catégories techniques à 15 % chacune (75 %), pitch/Q&A à 25 %. Le PDF ne fixe ni FPS minimal ni durée de vidéo. Il ne présente pas une vidéo comme remplacement de la démo live, et ne donne pas l’heure précise du code freeze.

L’histoire à raconter : **« Que devient une observation utile lorsque le rover voit encore, mais que Mission Control ne reçoit plus rien ? »**

| Catégorie du barème | Preuve à montrer dans Artemis-X | Limite à annoncer |
|---|---|---|
| Perception edge — 15 % | Webots RGB + AI DETECTION ; YOLO11n entraîné et figé, exécuté localement | Une classe apprise `rock`, données synthétiques ; CPU/GPU du laptop, pas un ordinateur spatial qualifié |
| Maths spatiales — 15 % | RGB/profondeur alignés → point de surface → transformation caméra/rover → odométrie ; coordonnées et incertitude dans une fiche de détection | Repère `odom` initialisé au départ ; pas de GPS, nord absolu ou recalage SLAM. Profondeur idéale de simulation |
| Télémétrie/protocole — 15 % | Paquets binaires, CRC, séquences ; presets réels DEGRADED/SEVERE/BLACKOUT/RESTORE ; ACK et retransmissions prioritaires | Transport borné, pas de garantie de livraison totale ; toutes les poses anciennes ne sont pas retransmises |
| Dashboard — 15 % | Carte reçue, fiches de détection, âge UTC, STALE au blackout, LIVE après reprise | Aucune vidéo downlink ; images caméra/IA/profondeur visibles dans Webots, pas inventées dans le navigateur |
| Intégration/space compliance — 15 % | Flux automatique de bout en bout ; files bornées ; provenance, validation, séparation des horloges et missions | Prototype de principes de résilience, pas une certification spatiale ; conduite scriptée sans évitement autonome |

Ne pas consacrer l’essentiel du temps à la sidebar : son rôle est de rendre les preuves lisibles. Le blackout et la reprise font démontrer plusieurs catégories simultanément.

## 2. Ce qu’il faut lancer

```text
Webots + caméra/profondeur → inférence YOLO locale → fusion/carte embarquée
  → télémétrie binaire + file DTN → émulateur UDP → FastAPI → MongoDB
                                                        → REST/WebSocket → Next.js
```

Le TCP localhost de l’inférence représente le calcul embarqué dans cette émulation ; il ne traverse pas la liaison lunaire. Le blackout coupe la télémétrie et les ACK, pas la perception locale. Le dashboard affiche seulement ce qui a atteint le récepteur sol.

### Vérification avant de commencer

Depuis la racine du dépôt, dans PowerShell :

```powershell
.\scripts\demo_atec.ps1 -Step check
```

Le script vérifie les environnements déjà installés, Node >= 24, Docker Desktop, les fichiers de configuration, le SHA-256 du modèle et tous les assets locaux référencés par le monde. `check` et `status` sont en lecture seule, sans démarrage ni changement de preset. Les ports affichés ne prouvent pas, à eux seuls, que le bon service fonctionne.

Sur cette machine lors de la préparation : MongoDB était **healthy** et le frontend écoutait déjà sur **3000** ; API **8000**, inférence **5761** et émulateur **5791/5792/5794** n’étaient pas démarrés. Revérifier au moment de filmer. Ne pas démarrer un deuxième frontend si celui en cours est bien le bon.

### Lancement, un service par terminal

Tous les terminaux partent de la racine du dépôt. Garder ceux de l’API, du réseau, de l’inférence et du frontend ouverts. Choisir un nom inédit de prise et utiliser le **même** pour `link` et `webots`, par exemple `prise-01`.

| Ordre | Commande | Attendre avant de poursuivre |
|---|---|---|
| 1. Base | `.\scripts\demo_atec.ps1 -Step database` | `docker compose -f docker/compose.yml ps mongodb` doit indiquer `healthy` |
| 2. API sol | `.\scripts\demo_atec.ps1 -Step ground` | Message de démarrage uvicorn ; health : MongoDB up, receiver enabled |
| 3. Réseau | `.\scripts\demo_atec.ps1 -Step link -RunName prise-01` | `[link] … preset NORMAL` ; chemin reverse ACK activé |
| 4. YOLO | `.\scripts\demo_atec.ps1 -Step inference` | Message JSON `listening: "127.0.0.1:5761"` après chargement/warmup ; device réellement utilisé |
| 5. Frontend | `.\scripts\demo_atec.ps1 -Step dashboard` | Next ready ; **sauter cette étape si le bon frontend est déjà lancé** |
| 6. Rover | `.\scripts\demo_atec.ps1 -Step webots -RunName prise-01` | Webots ouvre `artemis_r4_monitor.wbt` **en pause**, puis choisir **Real-time**, pas Fast |

Si l’inférence GPU ne fonctionne pas, utiliser `-Step inference -Device cpu`, puis répéter la prise et annoncer les performances CPU observées. Ne jamais présenter des mesures GPU comme des mesures CPU.

Le lanceur refuse les ports occupés et les sorties d’une prise déjà existante ; il n’arrête aucun processus et ne supprime rien. Il applique les variables de démo uniquement au service lancé et restaure l’environnement PowerShell ensuite. Il ne réentraîne rien, ne télécharge pas de modèle et ne modifie aucun fichier `.env`.

Pour vérifier après lancement :

```powershell
.\scripts\demo_atec.ps1 -Step status
(Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/health').data
(Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/network/status').data
```

Ports : frontend 3000 ; REST/WS 8000 ; inférence 5761 ; UDP rover→émulateur 5791→sol 5790 ; contrôle 5792 ; ACK sol→émulateur 5794→rover 5793. Ne pas lancer `scripts/ground_receiver.py` en parallèle du backend : ils se disputeraient 5790. Le runner de tests R6 utilise une autre convention d’inférence, 5771 ; ne pas mélanger ce runner avec cette prise.

### Préparer le navigateur et Webots

1. Ouvrir `http://localhost:3000/settings`. Mettre **Unified Mission Control OFF** pour les pages détaillées, et **Follow the newest mission ON**. Revenir à `http://localhost:3000/` sans ancien `?mission=…`.
2. Préparer les routes `/map`, `/detection`, `/network`, `/missions`, `/logs`. Préférer un seul onglet navigué via la sidebar : une mission épinglée dans un autre onglet peut créer une confusion visuelle.
3. Dans Webots, afficher les panneaux **RGB CAMERA, AI DETECTION, DEPTH SENSOR, LOCAL MAP 2D**. Si fermés : sélectionner le rover → **Overlays → Display Devices**. Le monde `artemis_test_course.wbt` n’a pas les quatre panneaux du monde monitor.
4. Démarrer en **Real-time** 20–30 secondes avant la prise et attendre de vraies réceptions : nouvelle mission, horloge rover qui avance, obstacles et header LIVE. Ne pas commencer le commentaire « live » tant que ces signes manquent.
5. Ne pas mettre Webots en pause pendant le blackout. Le but est de montrer la séparation entre calcul embarqué et réception sol.

La tour complète représente environ 8½ minutes de simulation dans les rapports existants ; le temps réel dépend de la machine. Une prise courte n’a pas besoin d’attendre les neuf zones, mais laisser ensuite la tour finir pour obtenir ses rapports finaux. Fermer immédiatement Webots peut empêcher l’écriture de `summary.json`.

### Installation, seulement si le preflight échoue

Faire cela avant le tournage, jamais pendant le pitch :

```powershell
uv sync --project backend --frozen
uv sync --project ground_station --frozen --extra dev
Push-Location frontend
pnpm install --frozen-lockfile
Pop-Location
```

Créer `docker/.env` et `ground_station/.env` depuis leurs `.env.example` **seulement s’ils n’existent pas**, puis configurer les identifiants MongoDB concordants sans les afficher à l’écran. Installer/ouvrir Docker Desktop et Webots R2025a. Les assets ne sont pas tous versionnés ; si absents :

```powershell
uv run --project backend --no-sync python -I artemis-webots/tools/fetch_polyhaven.py
```

Cette récupération nécessite Internet. Ouvrir une fois le monde avant la répétition : il référence aussi des ressources Webots distantes qui doivent être présentes/en cache. Ne pas promettre une installation entièrement hors ligne ; la **perception** elle-même n’utilise pas d’API cloud. `experiments/` n’est pas nécessaire à l’inférence : le runtime privilégie `models/artemis_rock_yolo11n.pt`. `artemis-webots/output/` sera recréé pour les nouvelles preuves ; ce n’est pas le modèle entraîné.

## 3. Vidéo de démo recommandée : environ 4 minutes

Ces durées sont un storyboard conseillé, pas une exigence de l’organisateur. Préparer la stack avant d’enregistrer ; ne pas filmer les installations. Avec deux personnes, une raconte et l’autre navigue/commande le réseau.

| Temps | Écran et geste | Texte à dire / preuve |
|---|---|---|
| 00:00–00:20 | Overview ; mission courante, header LIVE | « Artemis-X transforme des observations locales en informations spatiales utilisables au sol, même avec une liaison intermittente. Voici une mission réellement reçue. » |
| 00:20–00:55 | Webots ; RGB, AI DETECTION, DEPTH, carte embarquée | « Notre YOLO11n entraîné détecte les rochers localement. La profondeur alignée permet de projeter les pixels. Aucun appel cloud n’est nécessaire à cette perception. » |
| 00:55–01:25 | `/map`, puis `/detection` ; cliquer un obstacle `yolo+depth` | « Cette fiche montre l’identité, les coordonnées en mètres dans odom, la provenance, la confiance et l’incertitude. Ce sont des données reçues, pas la vérité terrain Webots. » Ne pas confondre confiance et précision géométrique. |
| 01:25–01:45 | `/network` ; appliquer DEGRADED, attendre confirmation | « Nous perdons maintenant 30 % des paquets avec 200 ms de délai configuré. Le sol montre ce qui arrive réellement. » Montrer pertes, débit et courbe, sans exiger exactement 30 % sur un petit échantillon. |
| 01:45–02:10 | Appliquer SEVERE ; conserver ~20 s | « Ici : 50 % de pertes, 1,3 s de délai et un plafond de 1 000 octets par seconde. Nous envoyons des informations structurées, pas le flux vidéo. » La médiane mesurée/cumulée peut différer du délai configuré. |
| 02:10–02:40 | BLACKOUT confirmé ; garder 15–20 s ; montrer carte puis Webots | « La réception s’arrête. La dernière carte reste visible, mais devient STALE et son âge augmente. À bord, les capteurs et la perception continuent. » Les compteurs de poses reçues doivent se stabiliser après les événements déjà en cours. |
| 02:40–03:20 | RESTORE confirmé ; attendre une réception, ouvrir map/logs | « La liaison est restaurée. Les observations importantes encore retenues sont retransmises et réconciliées par identité et temps d’observation. Le retour LIVE dépend d’une vraie réception. » Montrer l’âge qui baisse, les mises à jour et les trous de trajectoire, sans annoncer zéro perte. |
| 03:20–03:40 | `/missions` ; fiche d’une ancienne mission, puis revenir Follow newest | « Les missions sont persistées et isolées. Inspecter une archive ne la confond pas avec la session live. » Ne plus modifier les presets lorsqu’une ancienne mission est affichée. |
| 03:40–04:00 | Overview/map ; conclusion | « Nous démontrons perception, localisation, transmission compacte et visualisation honnête de la fraîcheur. Limites actuelles : domaine synthétique, odométrie et conduite scriptée. » |

Si RESTORE ne produit pas de réception dans le créneau prévu, continuer à attendre/diagnostiquer, pas annoncer une reprise fictive. Pour une vidéo montée, étiqueter clairement tout saut temporel. Pour le jury live, expliquer le délai observé et réduire la partie archive pour garder sept minutes.

### Cas supplémentaires pour répétition / Q&A

| Cas | Action | Résultat à vérifier |
|---|---|---|
| NORMAL | NORMAL pendant 20–30 s | Données fraîches, progression des réceptions ; aucune perte artificielle configurée |
| DEGRADED | 30 % de pertes, 200 ± 50 ms | Réceptions irrégulières, compteurs de pertes/retransmissions ; pas de faux lissage de position |
| SEVERE | 50 %, 1 300 ± 300 ms, 1 000 B/s | Stress sur délais/ACK ; ne pas promettre toutes les données. En real-time, le débit produit peut rester inférieur au plafond |
| BLACKOUT→RESTORE | BLACKOUT 15–20 s, puis RESTORE | STALE puis LIVE à la réception ; perception embarquée continue ; carte conservée, données critiques encore retenues récupérées |
| API interrompue | Ctrl+C dans **notre** terminal ground, attendre 5–10 s puis relancer `-Step ground` | DISCONNECTED, puis reconnexion WS/snapshot ; ne pas confondre cela avec un blackout radio où l’API reste connectée |
| Ancienne mission | Épingler une mission via `/missions`, puis Follow newest | Archive stable, pas de mélange d’identités avec la mission courante |
| Inférence interrompue | Hors prise principale, Ctrl+C dans **notre** terminal inference | Mapping profondeur continue ; état d’inférence indisponible signalé. Pas une garantie de navigation sûre |

Les contrôles réseau agissent sur l’émulateur **partagé/live**, même si une archive est affichée. Les presets sont confirmés par l’API ; un bouton cliqué n’est pas à lui seul la preuve que le changement est appliqué. Pas de panne MongoDB dans la démo principale : plus longue à expliquer et inutile pour prouver le blackout radio.

Commandes de secours dans un terminal séparé, si les boutons UI sont indisponibles :

```powershell
.\backend\.venv\Scripts\python.exe scripts/lunar_link.py ctl STATUS
.\backend\.venv\Scripts\python.exe scripts/lunar_link.py ctl DEGRADED
.\backend\.venv\Scripts\python.exe scripts/lunar_link.py ctl SEVERE
.\backend\.venv\Scripts\python.exe scripts/lunar_link.py ctl BLACKOUT
# Laisser 15–20 s en temps réel en observant les deux côtés.
.\backend\.venv\Scripts\python.exe scripts/lunar_link.py ctl RESTORE
```

Exécuter chaque changement au moment voulu, pas toute la liste en bloc. Ne pas réinitialiser le monde ou l’émulateur entre BLACKOUT et RESTORE : cela détruirait la continuité de la preuve.

## 4. Script du pitch officiel : 7 minutes

Préparer quatre supports simples : problème/architecture, maths, protocole, résultats/limites. La démo reste interactive ; la vidéo est une sauvegarde clairement annoncée, pas un substitut présenté comme live.

| Temps | Sujet | Message à défendre |
|---|---|---|
| 00:00–00:35 | Problème et solution | « Dans un cratère, le rover doit comprendre localement et le sol doit savoir ce qu’il ignore. Artemis-X privilégie une information spatiale utile, compacte et datée. » |
| 00:35–01:15 | Architecture | Montrer la chaîne logicielle ; calcul local distinct de la radio, files bornées, REST/WS sol ; préciser quelles bibliothèques sont réutilisées et quelles parties l’équipe a implémentées. |
| 01:15–02:05 | Perception + maths en live | Webots et fiche : détection → profondeur valide → transformation → repère odom ; annoncer les limites physiques et la provenance. |
| 02:05–02:40 | Efficacité du protocole | POSE = 33 octets ; obstacle = 26 octets + overhead du paquet ; CRC, séquence, horodatage, fragmentation bornée. Les tailles ne comprennent pas les en-têtes UDP/IP. |
| 02:40–03:15 | Réseau dégradé / sévère | Activer les presets réels, montrer pertes et plafond ; distinguer configuration du lien et mesures. |
| 03:15–03:45 | Blackout en live | Montrer STALE au sol et calcul continu à bord. Expliquer âge UTC et dernière observation en temps simulation. |
| 03:45–04:30 | Restore en live | Attendre la réception ; montrer réconciliation et événements reçus ; expliquer priorités, ACK et absence de garantie totale. |
| 04:30–05:10 | Dashboard et persistance | Carte et fiche informative ; une mission archivée puis retour au live ; aucune donnée de navigation inventée. |
| 05:10–06:10 | Résultats et tests | Deux ou trois chiffres avec environnement/source/date, et la sortie de tests réellement exécutés. Ne pas recycler un résultat GPU ou une capture historique comme mesure de la prise actuelle. |
| 06:10–07:00 | Limites et conclusion | « Nous avons réalisé la chaîne requise en émulation logicielle. La prochaine validation porte sur les capteurs bruités, la dérive et le calcul contraint ; l’évitement autonome n’est pas démontré ici. » |

### Explication mathématique courte

Dans le repère caméra **x avant, y gauche, z haut**, avec `d` la profondeur le long de l’axe optique :

```text
p_cam = (d, −(u−cx)·d/fx, −(v−cy)·d/fy)
p_rover = R_y(pitch) · p_cam + t_mount
p_odom_xy = R_z(yaw_est) · p_rover_xy + (x_est, y_est)
```

Le montage caméra est connu ; la pose estimée vient des encodeurs. Le prototype agrège des pixels de profondeur valides au premier plan, pas une distance arbitraire au centre de la boîte. Le point initial est un **point de surface visible**, pas le centre 3D exact du rocher ; fusion/tracking en déduisent ensuite une géométrie 2D. La vérité terrain du Supervisor sert aux évaluations, pas à alimenter la carte sol. Le repère de départ représente l’origine de référence du prototype, pas un site d’atterrissage géolocalisé.

### Réponses aux questions probables

- **Pourquoi pas envoyer les images ?** Coût radio ; la détection est locale et le sol reçoit coordonnées, identité, source, confiance, incertitude et états utiles. Le backend web peut servir du JSON sur localhost : ce n’est pas le format de la liaison lunaire.
- **Qu’est-ce qui est appris ?** Un détecteur de rochers YOLO11n. Projection, profondeur, odométrie, fusion, sérialisation et résilience sont du code déterministe. Les dépressions sont issues de la géométrie profondeur, pas d’une classe YOLO « crater » apprise.
- **Livrez-vous 100 % des données ?** Non. Les données critiques sont priorisées avec limites mémoire, TTL et essais ; les poses anciennes sont coalescées/fire-and-forget. Les pertes/pending sont comptées, pas cachées.
- **ACK = sauvegarde MongoDB ?** Non : ACK à réception sol. La file DB est en mémoire ; un crash du backend peut perdre des écritures déjà acquittées. La reprise réseau n’est pas une garantie de durabilité de bout en bout.
- **Quelle précision ?** Séparer précision simulée, dérive odométrique, incertitude et quantification binaire. Un pas de 1 mm dans le protocole ne signifie pas une localisation réelle à 1 mm.
- **Combien de FPS ?** Utiliser l’environnement et les mesures effectivement enregistrés. Cadence caméra, cadence de mapping, débit des réponses YOLO et paquets/s sont différents ; ne pas convertir naïvement une latence p50 en FPS réel.
- **Autonome / space compliant ?** Traitement local et transport résilient démontrés ; parcours de test scripté, pas de boucle d’évitement. Principes de robustesse illustrés, pas de certification spatiale.

## 5. Comment filmer proprement

Proposition de réglages : **1920×1080, 30 fps**, navigateur à 100 % si lisible, microphone clair, notifications désactivées. Tester une courte prise : OBS et YOLO peuvent se partager le GPU, donc vérifier la charge et les latences avant le live. Ne pas montrer les `.env`, mots de passe, URI authentifiées, onglets privés ou identifiants inutiles.

Dans OBS, créer deux scènes : **Mission Control** (capture de fenêtre navigateur) et **Onboard/Webots** (capture de fenêtre Webots). Une troisième scène côte à côte est utile au blackout si les panneaux et l’âge restent lisibles. Garder chaque source en ratio correct et vérifier que les overlays Webots sont capturés. Configurer le micro et faire un test d’enregistrement avant la prise, comme le recommande le [guide de démarrage officiel OBS](https://obsproject.com/kb/quick-start-guide).

Enregistrer en **MKV**, puis convertir sans réencodage via **File → Remux Recordings** vers MP4 : OBS recommande MKV pour limiter les dégâts d’une interruption brutale. Voir le [guide d’enregistrement officiel OBS](https://obsproject.com/kb/standard-recording-output-guide).

Garder le segment **BLACKOUT → RESTORE** en continu, avec même mission et compteurs visibles ; éviter les cuts qui empêchent de vérifier la causalité. Les captures [R6 existantes](screenshots/r6/) sont des preuves historiques, pas le flux du jour. Les nommer comme telles si utilisées en secours.

## 6. Preuves à garder et tests pour la livraison

Le lanceur écrit sous `artemis-webots/output/<RunName>/` (git-ignored), sans effacer les anciennes prises :

- `link.jsonl` : changements de preset, transmissions/pertes réelles ; `link-stats.json` à l’arrêt propre du réseau.
- `monitor-panels/` : captures embarquées aux temps simulation 20 et 45 s.
- `tour/summary.json` : rapport final après la fin du parcours ; stats inférence/mapping et `telemetry.dtn` (ACK, retries, queue, pending).
- `tour/inference/detections.jsonl` et `tour/mapping/obstacles_final.json` : perception et export final.
- MongoDB : missions, poses, obstacles, événements et mesures effectivement reçus ; le navigateur ne remplace pas ces preuves.

Après la prise, laisser le parcours finir. Remettre **RESTORE**, quitter Webots normalement, puis Ctrl+C dans les terminaux réseau, inférence et API que l’on a lancés. Ne pas tuer des processus préexistants. La base peut rester démarrée ; ne jamais utiliser `docker compose down -v` pour cette préparation.

Avant le code freeze, exporter une copie limitée des preuves finales dans le livrable (en revoyant chemins/secrets) : vidéo, résumé de la prise, sorties de tests, sources des chiffres, identifiant de révision et guide de lancement. Ne pas supprimer à nouveau `output/` avant cette exportation. Le PDF demande les sorties automatisées, pas seulement un screenshot vert ou un rapport affirmant « PASS ».

Tests ciblés, depuis la racine ; ils ne relancent pas l’entraînement ni le test ML verrouillé :

```powershell
# Protocole et transport : sockets de test dédiés / fixtures.
.\backend\.venv\Scripts\python.exe -m unittest tests.test_telemetry tests.test_telemetry_transport tests.test_telemetry_emulator tests.test_telemetry_dtn -v

# API : utilise MongoDB et des bases de test temporaires, supprimées par les tests.
uv run --project ground_station --no-sync python -m unittest discover -s ground_station/tests -v

Push-Location frontend
pnpm --filter web test
pnpm --filter web check-types
pnpm --filter web build
Pop-Location
```

Enregistrer leurs sorties dans des fichiers de livraison après exécution et vérifier les codes de sortie ; un test SKIPPED n’est pas une réussite. Pour une validation complète, **hors prise**, frontend et MongoDB disponibles mais API/émulateur/inférence/Webots manuels arrêtés :

```powershell
uv run --project ground_station --no-sync python scripts/r6_dashboard_e2e.py --out artemis-webots/output/validation-prise-01
```

Ce runner démarre/arrête ses propres services, pilote les presets et teste une interruption du backend. Il peut changer les préférences du navigateur de test et crée une vraie mission ; ne pas le lancer pendant le pitch. Consulter sa sortie récente, pas présumer qu’un rapport historique est encore valable.

Vérifications effectuées pendant cette préparation : syntaxe PowerShell valide ; `-Step check` réussi ; refus d’un deuxième frontend sur le port 3000 sans arrêter celui en cours ; **48/48 tests protocole/UDP/émulateur/DTN réussis en 20,767 s**. Ces tests utilisent leurs propres fixtures et sockets : ils ne constituent pas une nouvelle prise live Webots→dashboard. Ni test ML verrouillé ni entraînement relancé.

### Chiffres historiques utilisables avec leur contexte

- **Modèle synthétique** : AP50 0,777, rappel 0,774 ; annotations automatisées non vérifiées humainement — [model card](../models/model_card.md). Ce n’est ni un taux de reconnaissance parfait ni une validation lunaire réelle.
- **Sérialisation R5.1** : 11 622 octets binaires contre 85 721 octets de JSON équivalent sur 16 exports, soit 86,4 % de réduction, hors UDP/IP — [rapport protocole](TELEMETRY_R5_1_PROTOCOL.md). Le benchmark est historique, pas recalculé ici.
- **Résilience R5.4** : succès des observations critiques dans certains blackouts testés, mais pertes/pending et carte incomplète dans le stress SEVERE — [rapport DTN](TELEMETRY_R5_4_STORE_AND_FORWARD.md). Les anciens tests en Fast ne prédisent pas un délai garanti en Real-time.
- **Chaîne sol R5.5 et dashboard R6** : validations réelles documentées, séparation des horloges et états STALE/DISCONNECTED — [backend](TELEMETRY_R5_5_GROUND_BACKEND.md), [dashboard](DASHBOARD_R6_MISSION_CONTROL.md), [pages détaillées](DASHBOARD_R6_1_WORKSPACES.md). Les chemins de runs mentionnés peuvent avoir disparu avec le nettoyage ; conserver/produire des preuves actuelles.

Source comportement actuelle : `src/artemis_x/sim/rover.py:run_tour` applique les commandes calculées sur odométrie sans décision moteur liée aux détections. Les vieux documents de statut décrivent parfois des étapes désormais intégrées ; pour la démo, se fier au code actuel et aux rapports R5/R6, pas à une ancienne matrice de roadmap.
