// Relief 3D du portefeuille : une colonne par projet, dont la hauteur vaut le score.
// Les colonnes sont rangées en spirale par rang (le n° 1 au centre). Quand les poids
// changent, Python envoie le nouveau classement et chaque colonne glisse vers sa
// nouvelle place et sa nouvelle hauteur.
//
// Dialogue avec Streamlit (composant maison, sans bibliothèque) :
// - Streamlit envoie {type: "streamlit:render", args: {projets, hauteur, etat}} ;
// - on répond par des messages {isStreamlitMessage: true, type: ...} pour
//   annoncer qu'on est prêt, fixer la hauteur et renvoyer le projet cliqué.
import * as THREE from "three";
import { OrbitControls } from "./OrbitControls.js";

const COULEURS = { papier: 0xfae1cf, sol: 0xeecbb4, encre: 0x4a423c, bleu: 0x1d4f8c, bordeaux: 0x8e1b3a, survol: 0xc9853f };
const ECART = 1.25;          // distance entre deux colonnes
const HAUTEUR_MAX = 7;       // hauteur d'une colonne de score 1
const HAUTEUR_REPOS = 0.12;  // colonne sans score (avant l'analyse)
const mouvementReduit = matchMedia("(prefers-reduced-motion: reduce)").matches;

const zone = document.getElementById("scene");
const bulle = document.getElementById("bulle");
const etat = document.getElementById("etat");

function envoyer(type, donnees = {}) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...donnees }, "*");
}

// --- Repli sans WebGL ----------------------------------------------------------
let rendu;
try {
  rendu = new THREE.WebGLRenderer({ antialias: true });
} catch (erreur) {
  zone.innerHTML = '<p class="repli">Ce navigateur ne permet pas la 3D (WebGL). Le classement reste lisible dans les onglets.</p>';
  envoyer("streamlit:setFrameHeight", { height: 80 });
}

if (rendu) {
  rendu.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  rendu.shadowMap.enabled = true;
  rendu.shadowMap.type = THREE.PCFSoftShadowMap;
  zone.appendChild(rendu.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(COULEURS.papier);
  scene.fog = new THREE.Fog(COULEURS.papier, 28, 70);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
  camera.position.set(14, 13, 22);
  const controles = new OrbitControls(camera, rendu.domElement);
  controles.enableDamping = true;
  controles.maxPolarAngle = Math.PI * 0.47;  // jamais sous le sol
  controles.minDistance = 8;
  controles.maxDistance = 60;
  controles.autoRotate = !mouvementReduit;
  controles.autoRotateSpeed = 0.35;
  // L'utilisateur prend la main : la rotation automatique s'arrête, puis reprend après 8 s d'inactivité.
  let reprise;
  controles.addEventListener("start", () => { controles.autoRotate = false; clearTimeout(reprise); });
  controles.addEventListener("end", () => {
    if (!mouvementReduit) reprise = setTimeout(() => (controles.autoRotate = true), 8000);
  });

  scene.add(new THREE.HemisphereLight(0xfff6ee, 0xd9b9a0, 1.6));
  const soleil = new THREE.DirectionalLight(0xffffff, 1.6);
  soleil.position.set(-12, 24, 14);
  soleil.castShadow = true;
  soleil.shadow.mapSize.set(1024, 1024);
  Object.assign(soleil.shadow.camera, { left: -30, right: 30, top: 30, bottom: -30 });
  scene.add(soleil);

  const sol = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({ color: COULEURS.sol, roughness: 1 }));
  sol.rotation.x = -Math.PI / 2;
  sol.receiveShadow = true;
  scene.add(sol);

  // Une seule géométrie partagée par toutes les colonnes (InstancedMesh) : fluide même avec 2 000 projets.
  const geometrie = new THREE.BoxGeometry(0.82, 1, 0.82).translate(0, 0.5, 0);
  const matiere = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.75, metalness: 0 });
  let colonnes = null;
  let projets = [];
  let etats = [];            // position et hauteur actuelles de chaque colonne
  let survol = -1;
  const etiquettes = [];     // numéros des trois projets recommandés, en HTML au-dessus des colonnes
  const outil = new THREE.Object3D();
  const couleur = new THREE.Color();

  function couleurDe(projet, i) {
    if (i === survol) return COULEURS.survol;
    if (projet.anomalie) return COULEURS.bordeaux;
    if (projet.reco) return COULEURS.bleu;
    return COULEURS.encre;
  }

  // Disposition en spirale (comme les graines d'un tournesol) : le n° 1 au centre, puis chaque rang
  // un peu plus loin. Les meilleurs projets forment le sommet du relief.
  const ANGLE_OR = Math.PI * (3 - Math.sqrt(5));
  function cible(projet, bornes) {
    const r = projet.rang - 1;
    const rayon = ECART * 0.58 * Math.sqrt(r + 0.5);
    let h = HAUTEUR_REPOS;
    if (projet.score != null) {
      // Hauteur relative, du score le plus faible au plus fort du portefeuille, accentuée vers le haut :
      // les meilleurs projets se détachent en sommet au lieu de former un plateau.
      const etendue = bornes.max - bornes.min || 1;
      h = 0.2 + ((projet.score - bornes.min) / etendue) ** 2.2 * HAUTEUR_MAX;
    }
    return { x: rayon * Math.cos(r * ANGLE_OR), z: rayon * Math.sin(r * ANGLE_OR), h };
  }

  // Cadre la caméra sur le relief entier ; refait seulement quand le nombre de projets change,
  // pour ne pas défaire le point de vue choisi par l'utilisateur à chaque mouvement de curseur.
  let distanceIdeale = 0;
  function cadrer(nombre) {
    const rayon = ECART * 0.58 * Math.sqrt(nombre);
    controles.target.set(0, 1.5, 0);
    distanceIdeale = rayon * 2.3 + 10;
    scene.fog.near = distanceIdeale * 0.8;   // le fond se fond dans le papier, l'avant reste net
    scene.fog.far = distanceIdeale * 2.4;
    controles.maxDistance = distanceIdeale * 2;
    redimensionner();
  }

  function recevoir(nouveaux) {
    const anciens = new Map(projets.map((p, i) => [p.id, etats[i]]));
    projets = nouveaux;
    if (!colonnes || colonnes.count < projets.length) {
      if (colonnes) scene.remove(colonnes);
      colonnes = new THREE.InstancedMesh(geometrie, matiere, Math.max(projets.length, 1));
      colonnes.castShadow = true;
      colonnes.receiveShadow = true;
      scene.add(colonnes);
    }
    if (projets.length !== colonnes.count || !distanceIdeale) cadrer(projets.length);
    const scores = projets.map((p) => p.score).filter((s) => s != null);
    const bornes = { min: Math.min(...scores), max: Math.max(...scores) };
    colonnes.count = projets.length;
    etats = projets.map((p, i) => {
      const but = cible(p, bornes);
      // Une colonne déjà présente part de sa place actuelle ; une nouvelle sort du sol.
      const depart = anciens.get(p.id) || { x: but.x, z: but.z, h: 0.01 };
      // Petit décalage selon le rang : les colonnes se mettent en place en vague, du centre vers le bord.
      return { ...depart, but, delai: mouvementReduit ? 0 : Math.min(p.rang * 6, 900) };
    });
    if (mouvementReduit) etats.forEach((e) => Object.assign(e, e.but));
    debutTransition = performance.now();
    survol = -1;
    colorier();
    placerEtiquettes();
  }

  function colorier() {
    projets.forEach((p, i) => colonnes.setColorAt(i, couleur.setHex(couleurDe(p, i))));
    if (colonnes.instanceColor) colonnes.instanceColor.needsUpdate = true;
  }

  function placerEtiquettes() {
    etiquettes.forEach((e) => e.element.remove());
    etiquettes.length = 0;
    projets.forEach((p, i) => {
      if (!p.reco || p.anomalie) return;
      const element = document.createElement("div");
      element.className = "etiquette";
      element.textContent = `n° ${p.numero ?? p.rang} · ${p.id}`;  // textContent : jamais lu comme du HTML
      document.body.appendChild(element);
      etiquettes.push({ element, i });
    });
  }

  // --- Animation -------------------------------------------------------------
  let debutTransition = 0;
  let avant = performance.now();
  function image(maintenant) {
    const dt = Math.min((maintenant - avant) / 1000, 0.05);
    avant = maintenant;
    // Lissage exponentiel : rapide au départ, doux à l'arrivée.
    const k = 1 - Math.exp(-dt * 5);
    if (colonnes) {
      etats.forEach((e, i) => {
        if (maintenant - debutTransition > e.delai) {
          e.x += (e.but.x - e.x) * k;
          e.z += (e.but.z - e.z) * k;
          e.h += (e.but.h - e.h) * k;
        }
        outil.position.set(e.x, 0, e.z);
        outil.scale.set(1, Math.max(e.h, 0.01), 1);
        outil.updateMatrix();
        colonnes.setMatrixAt(i, outil.matrix);
      });
      colonnes.instanceMatrix.needsUpdate = true;
    }
    controles.update();
    rendu.render(scene, camera);
    // Les étiquettes suivent le haut de leur colonne à l'écran.
    const largeur = zone.clientWidth, hauteur = zone.clientHeight;
    const places = etiquettes.map(({ element, i }) => {
      const e = etats[i];
      const point = new THREE.Vector3(e.x, e.h + 0.4, e.z).project(camera);
      return { element, x: ((point.x + 1) / 2) * largeur, y: ((1 - point.y) / 2) * hauteur, visible: point.z < 1 };
    });
    // Deux étiquettes trop proches : la plus haute remonte encore d'une ligne, pour ne jamais se chevaucher.
    places.sort((a, b) => b.y - a.y).forEach((place, j) => {
      places.slice(0, j).forEach((autre) => {
        if (Math.abs(place.x - autre.x) < 120 && Math.abs(place.y - autre.y) < 22) place.y = autre.y - 22;
      });
      place.element.style.left = `${place.x}px`;
      place.element.style.top = `${place.y}px`;
      place.element.style.display = place.visible ? "block" : "none";
    });
    requestAnimationFrame(image);
  }
  requestAnimationFrame(image);

  // --- Survol et clic --------------------------------------------------------
  const rayon = new THREE.Raycaster();
  const souris = new THREE.Vector2();
  function viser(evenement) {
    const cadre = rendu.domElement.getBoundingClientRect();
    souris.set(((evenement.clientX - cadre.left) / cadre.width) * 2 - 1, -((evenement.clientY - cadre.top) / cadre.height) * 2 + 1);
    rayon.setFromCamera(souris, camera);
    const touche = colonnes ? rayon.intersectObject(colonnes)[0] : null;
    return touche ? touche.instanceId : -1;
  }

  function lignesBulle(p) {
    const lignes = [`${p.id} · ${p.nom}`, `${p.secteur} · ${p.pays}`];
    if (p.score != null) {
      const rang = p.anomalie ? "écarté" : `Rang ${p.numero ?? p.rang}`;  // les dossiers anormaux n'ont pas de rang
      lignes.push(`${rang} · score ${p.score.toFixed(3).replace(".", ",")}`);
    }
    if (p.anomalie) lignes.push("Dossier anormal : écarté de la recommandation");
    return lignes;
  }

  rendu.domElement.addEventListener("pointermove", (evenement) => {
    const i = viser(evenement);
    if (i !== survol) { survol = i; colorier(); }
    if (i < 0) { bulle.classList.remove("visible"); return; }
    bulle.replaceChildren(...lignesBulle(projets[i]).map((texte, n) => {
      const element = document.createElement(n ? "span" : "b");
      element.textContent = texte;
      return element;
    }));
    const x = Math.min(evenement.clientX + 14, zone.clientWidth - bulle.offsetWidth - 8);
    const y = Math.min(evenement.clientY + 14, zone.clientHeight - bulle.offsetHeight - 8);
    bulle.style.left = `${Math.max(8, x)}px`;
    bulle.style.top = `${Math.max(8, y)}px`;
    bulle.classList.add("visible");
  });
  rendu.domElement.addEventListener("pointerleave", () => {
    survol = -1; colorier(); bulle.classList.remove("visible");
  });
  // Un clic (sans glisser) envoie l'identifiant du projet à Python, qui affiche sa fiche.
  let appui = null;
  rendu.domElement.addEventListener("pointerdown", (e) => (appui = [e.clientX, e.clientY]));
  rendu.domElement.addEventListener("pointerup", (evenement) => {
    if (!appui || Math.hypot(evenement.clientX - appui[0], evenement.clientY - appui[1]) > 5) return;
    const i = viser(evenement);
    if (i >= 0) envoyer("streamlit:setComponentValue", { value: projets[i].id, dataType: "json" });
  });

  // --- Taille ----------------------------------------------------------------
  function redimensionner() {
    rendu.setSize(zone.clientWidth, zone.clientHeight);
    camera.aspect = zone.clientWidth / Math.max(zone.clientHeight, 1);
    // Écran étroit : la caméra recule pour garder tout le relief dans le cadre.
    if (distanceIdeale) {
      const direction = camera.position.clone().sub(controles.target).normalize();
      const distance = distanceIdeale * (zone.clientWidth < 520 ? 1.45 : 1);
      camera.position.copy(controles.target).addScaledVector(direction, distance);
    }
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(redimensionner).observe(zone);

  // --- Messages de Streamlit -------------------------------------------------
  let hauteurCadre = 0;
  window.addEventListener("message", (evenement) => {
    if (evenement.data?.type !== "streamlit:render") return;
    const args = evenement.data.args || {};
    if (args.hauteur !== hauteurCadre) {
      hauteurCadre = args.hauteur;
      envoyer("streamlit:setFrameHeight", { height: hauteurCadre });
    }
    etat.textContent = args.etat || "";
    recevoir(args.projets || []);
  });
}

envoyer("streamlit:componentReady", { apiVersion: 1 });
