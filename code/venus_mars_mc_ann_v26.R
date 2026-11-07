# =============================================================================
# venus_mars_mc_ann_v26.R
# Venus cloud-level aerostat colony vs. Mars surface colony
# Monte Carlo simulation wrapped around an H2O deep learning (ANN) regression
#
# Changes from v25
#   * Call-level retries for H2O (v25 failed RQ2_city three times). The H2O
#     logs show the JVM healthy throughout: on the third attempt a model
#     finished at 06:12:57, the R client's next request never reached the
#     server, and 22 s later the same server answered the restart request
#     normally. The failures are brief client-side connection drops, a few
#     per hour of traffic, while RQ2_city needs several hours of continuous
#     traffic, so restarting the whole run after a drop could never finish.
#     Every H2O operation (upload, fit, predict, delete) is now retried in
#     place up to 6 times with growing waits (10-60 s) while the JVM is up;
#     only if the JVM is actually down does the run restart. Every error and
#     retry is time-stamped to <out_dir>/h2o_errors.log. Model and
#     statistics are unchanged from v24/v25.
#
# Changes from v24 (v25)
#   * H2O stability fixes after the v23 run failed RQ2_city three times with
#     "Unexpected CURL error: Timeout was reached ... Connection timed out
#     after 10007 milliseconds". The H2O logs showed no errors and no new JVM
#     after the retries, so the same JVM stayed up but stopped accepting
#     connections in time, and the v23 retry never actually restarted it.
#     Model and statistics are unchanged from v24.
#     - Prediction frames are now removed after every predict (v15-v24 left
#       one H2O frame behind per prediction, thousands per run).
#     - The H2O client's connection timeout is raised from the curl package
#       default of 10 s to 120 s, so a busy or pausing JVM is waited for
#       instead of failing the run; any single request that hangs for 15
#       minutes now fails (previously it could wait forever) and triggers a
#       restart.
#     - A fresh H2O JVM is started before every run, so nothing accumulates
#       across runs.
#     - After any failure the JVM is restarted for real: shutdown is
#       requested, and if the port is still held the java process listening
#       on it is ended (found via netstat on Windows, lsof elsewhere).
#     - H2O logs at INFO level (records memory and GC pressure).
#
# Changes from v23 (v24)
#   * Venus cloud-level weather (author's decision, 2026-10-16).
#     - Habitat altitude now 54-60 km (was 50-55). The main convective layer
#       (cells about 20 km across, updrafts about 2 m/s, downdrafts to about
#       3 m/s; VEGA balloons logged downdrafts to about 3.5 m/s at 53.6 km)
#       tops out at 55.5-59 km depending on source (model about 55.5; VeRa
#       49-59; Akatsuki 50-58), and the strong upper-cloud shear (wind rising
#       from about 62 to 102 m/s) lies above about 62 km. The convective-layer
#       top is drawn 55.5-59 km; inside it the hull carries a gust and
#       altitude-control surcharge of 3-12% of capital, above it 30% of that.
#     - Thinner air aloft needs more envelope: surcharge = envelope share of
#       capital (5-20%) x (density at 52.5 km / density at habitat - 1).
#     - Temperature and pressure now follow the VIRA reference atmosphere
#       (45-65 km) instead of a 50-55 km linear fit. Above about 54 km the
#       habitat is colder than 22 C, so Venus now pays a heating load,
#       computed the same way as Mars.
#     - Radiation (RQ1) uses the same 54-60 km band, so the Venus dose rises
#       with altitude (1.45-1.55 x per km above 51 km).
#     - Drift latitude capped at 50 deg (circulation changes sharply near
#       55 deg; polar cold collar at about 60-70 deg).
#     - Collector section: relative shear wind 2-20 m/s (Akatsuki 2016
#       equatorial jet to 80 m/s against about 60 m/s background) and tether
#       structural factor 1.1-1.5.
#     - The planetary-scale lower-cloud disruption (47.5-56.5 km, up to
#       7,500 km wide, about 328 km/h) interrupts 0-10% of collection time
#       and, for a habitat below 56.5 km, 0-5% of sunlight.
#
# Changes from v22 (v23)
#   * Run robustness after the v22 run lost its H2O connection during RQ1
#     ("H2O connection has been severed ... timed out after 10006 ms").
#     Model, statistics and network settings are unchanged from v22.
#     - H2O uses all cores but two (leaves headroom for Windows and the R
#       client, so the JVM keeps answering health checks) and a 16 GB heap
#       (ample for 12,000-row frames, leaves more RAM for the OS).
#     - H2O logs are written to <out_dir>/h2o_logs so a JVM crash leaves a
#       record of its cause.
#     - Each run is retried up to 3 times; if the cluster is down it is
#       restarted, and the run repeats from its start with the same seed, so
#       results are identical to an uninterrupted run.
#     - Each finished run's summary row is saved to <out_dir>/checkpoints.
#       Rerunning the whole script skips runs already finished and picks up
#       at the first unfinished one (delete the checkpoints folder to force a
#       full rerun).
#     - The H2O object store is cleared between runs.
#
# Changes from v21 (v22)
#   * Transport economies of scale (author's decision, 2026-10-15). The
#     tether fee and the mass-driver station capital charge are no longer
#     fixed per-kg draws; each is capital divided by annual throughput, with
#     throughput set by colony size x two-way traffic per colonist (imports
#     and exports share the same tethers and drivers).
#     - Tether: facility mass 60-200 x payload (HASTOL Phase I about 160-200 x
#       a 14 t payload; Phase II tether about 3 x lighter), delivered at the
#       planet's transport cost x 0.3-1.2 (local ballast cheaper), annual
#       capital charge 8-20%, 20-100 catches per year; reference capacity
#       14 t x catches per year.
#     - Station: $50-2,000/kg at a reference 1,000 t/yr.
#     - Both scale with throughput by the cost-capacity rule: capital ~
#       capacity^b, b 0.6-0.9, so cost per kg ~ (throughput / reference)^(b-1).
#   * Applies to RQ3 (all scenarios) and RQ4 (both planets, each at its own
#     transport cost and the same colony traffic).
#
# Changes from v20 (v21)
#   * RQ3 return architecture (author's plan, 2026-10-15): cargo aerocaptures
#     into Earth orbit rather than entering directly, and ships in longer
#     containers/sabots. Capsule product share now follows from container
#     length-to-diameter ratio (shield mass scales with frontal area, product
#     with volume): share = L/D / (L/D + c), L/D 1-4, c 0.2-0.6 [A], less the
#     entry-speed penalty. This replaces entry_payload_frac and
#     aerocapture_frac in all RQ3 scenarios.
#   * Earth-market goods (rocket, full, He-3) then descend from orbit as
#     consolidated downmass ($50-10,000/kg [S/A]: backhaul on reusable Earth
#     shuttles at the low end, mature dedicated return market about
#     $5,000-10,000/kg at the high end; Dragon CRS today about $63,000/kg),
#     replacing the direct-entry recovery fee. Cislunar buyers (protein, D2O)
#     take delivery in orbit with no descent cost.
#
# Changes from v19 (v20)
#   * RQ4 research-contract income (author's decision, 2026-10-15), on the
#     Antarctic-station model: a government research program budget per
#     colony, spread over the colony population, times a planet-specific
#     science-value multiplier drawn from the same range on both planets.
#     Venus claims: 1g space-medicine control lab, extreme-environment
#     (surface) engineering with low-latency teleoperation, solar-rich
#     propulsion test bed, runaway-greenhouse and cloud-chemistry science.
#     Mars claims: subsurface life search, polar ice record, accessible
#     geology. Equal ranges, so the term adds no built-in advantage.
#
# Changes from v18 (v19)
#   * RQ4 He-3 market depth (author's decision, 2026-10-15). Present demand
#     is only a few kg per year (federal < 6,000 L/yr, about 0.8 kg; Bluefors
#     up to 10,000 L/yr, about 1.35 kg). Colony-wide supply (per-colonist
#     output x colony size) meets a market of 1-20 kg/yr, less the share held
#     by lunar and terrestrial tritium-decay supply, with constant-elasticity
#     price response; below unit elasticity the colony caps sales at its share
#     of the market and processes only the air that requires.
#   * RQ4 national-security cost share of startup capital (0-30%), paid under
#     the same rule on both planets: only to a colony that supplies He-3 or
#     heavy water at a profit (Mars qualifies through heavy water alone).
#   * RQ3 isotope scenario unchanged: it remains a first-tonne margin test at
#     the prevailing price.
#
# Changes from v17 (v18)
#   * RQ4 settlement-motive terms (author's decision, 2026-10-15):
#     - Quality-of-life hardship premium on colonist compensation, anchored on
#       US post hardship and danger pay scales (each 5-35% of base pay).
#       Mars sits higher because every step outside needs a pressure suit and
#       the trip is longer; Venus keeps a premium for acid aerosol and the
#       lethal surface below. Gravity is excluded here (already in RQ5).
#     - Bulk supply line (the "Oregon" role): each colony sells its cheaper
#       volatile to cislunar buyers at Earth-launch parity, Venus nitrogen
#       from its air and Mars water from its ice, shipped at each planet's own
#       export cost, so Mars's shallower gravity well counts in its favor.
#   * Reported only (not in D): capital per colonist, the colonist resource
#     claim value (PV per colonist of co-product, supply and export streams,
#     the Donation Land Claim analog), and the cost cut needed to bring
#     capital per colonist to about three years of colonist compensation.
#
# Changes from v16 (v17)
#   * Helium-3 and deuterium (heavy water, D2O) enter as co-products of
#     sulfuric acid mining (author's decision, 2026-10-15).
#     - Deuterium comes out of the acid-derived water itself. Venus water has
#       D/H about 0.016-0.025, 100-150 times Earth's 1.56e-4, so far less feed
#       is processed per kg D2O. Mars ice (about 5-8 times Earth) gets the same
#       treatment for fairness.
#     - Helium-3 is in the gas phase, so it is recovered from the air stream
#       the acid collector already pulls through its mesh pads. Venus air holds
#       about 12 ppm He; the Venus 3He/4He ratio is unmeasured (Pioneer Venus
#       upper limit 3e-4), so it spans Earth air (1.4e-6) to that limit. Mars
#       gets no He-3 credit (no measured He resource in 6 mbar air).
#   * New RQ3 scenario "isotope": net co-product margin per tonne of
#     acid-derived water processed. He-3 ships to Earth by the full launch
#     chain; D2O is sold in cislunar space against Earth heavy water plus
#     Earth launch (same market structure as the protein scenario).
#   * RQ4: co-product revenue per colonist-year on both planets, each line run
#     only when its margin is positive (operator's choice inside the NPV).
#
# Changes from v15 (v16)
#   * Startup check for a stale H2O cluster: if an H2O instance from a
#     previous session is still listening on localhost:54321, the script
#     connects, shuts it down, waits until the port is released, and then
#     starts a fresh cluster with this script's thread and memory settings.
#     Model physics, statistics and network settings are unchanged from v15.
#
# Changes from v14 (v15)
#   * Categorical encoding back to one-hot ("OneHotInternal"), keeping
#     use_all_factor_levels = TRUE (author's decision after the v14 benchmark).
#   * Activation RectifierWithDropout (author's choice) with hidden dropout
#     0.05 per layer. Full-scale benchmark (12,000 rows, 64x32, 40 epochs,
#     holdout R2), dropout 0.5 (H2O default) / 0.2 / 0.1 / 0.05 vs plain
#     Rectifier: RQ4 0.843 / 0.924 / 0.946 / 0.948 vs 0.939; RQ3 protein
#     0.829 / 0.948 / 0.979 / 0.990 vs 0.990; RQ3 full 0.902 / 0.953 / 0.978 /
#     0.986 vs 0.985; RQ2 0.853 / 0.928 / 0.944 / 0.939 vs 0.937.
#
# Changes from v13
#   * Categorical inputs enter the ANN as true factors: RQ3 product identity
#     (rocket, full) and RQ3 protein market (LEO/cislunar vs Mars orbit,
#     replacing the 0-1 market_u number). H2O options:
#     categorical_encoding = "Eigen" and use_all_factor_levels = TRUE.
#   * Factor sensitivities are counterfactual: per prediction, the mean
#     absolute change in prediction when the factor is switched to each other
#     level (unsigned; the numeric inputs keep +/- 0.1 SD central differences).
#   * Activation is a CFG choice; see CFG$activation for the benchmarked pick.
#   * Export-product inputs (price, biomass, extraction, compliance, premium,
#     Earth recovery) are tagged "export" and no longer enter the protein
#     scenario as noise inputs.
#
# Changes from v12
#   * RQ3 second scenario ("protein"): Venus gas-fermentation microbial protein
#     (hydrogen-oxidizing bacteria, Solar Foods Solein principle) sold to
#     off-Earth markets, low Earth orbit / cislunar or Mars orbit, against
#     supply from Earth. Inputs on Venus: ambient CO2, H2 from acid-derived
#     water, O2 coproduct, N2 from the air via ammonia, sulfur from the acid,
#     station electricity. Energy 18-85 kWh per kg protein (thermodynamic
#     floor to about 62.5 kWh/kg estimated for Factory 01; Smaje 2024). Earth
#     supply: protein cost + Earth launch ($100/kg Starship-class target to
#     $7,000/kg SpaceX rideshare, Feb 2026) + mass-driver departure from LEO
#     for Mars. Both suppliers aerocapture at the destination. For this
#     scenario D = Earth-supplied cost minus Venus-delivered cost.
#   * RQ4 protein module (both planets): gas fermentation replaces imported
#     protein beyond what crops supply; electricity at each colony's derived
#     price, bioreactor plant mass imported, nitrogen at each planet's cost,
#     Mars CO2 compression from 6 mbar.
#
# Changes from v11
#   * RQ3 station electricity is derived, not assumed: hydrogen is made in
#     daylight on Venus arrays, so USD/kWh = imported array mass per average W
#     (from the Venus power physics) x Earth-to-Venus transport / lifetime hours.
#   * Minor-cannabinoid isolates split into CBG (wholesale about $1,250/kg,
#     KV Cannabis 2026) and THCV (Red Mesa Science: $832 per 100 g to about
#     $4,875/kg). These replace the earlier $5k-30k assumption.
#   * Paclitaxel biomass per kg tied to cell-culture titers: 28-110 mg/L with
#     methyl jasmonate elicitation (PMC10673093), at 10-25 g/L dry cell mass,
#     gives about 90-900 kg dry cells per kg (bark: 10,000 kg leaves + bark).
#
# Changes from v10
#   * Shuttle hardware split three ways (RQ3 and RQ4, both planets): carbon-
#     composite airframe built locally from atmospheric CO2 carbon (local
#     fabrication cost), imported avionics/precision parts, and the NTR
#     reactor, which is imported and replaced on its own flight life (NRX-A6
#     ran 62 min continuously in 1967; an ascent burn is minutes). Airframe
#     reuse 100-300 flights with local maintenance (author's design target
#     about 200; Falcon 9 booster record about 37 by 2026).
#   * Mass-driver departure derived from physics on both planets, replacing
#     flat fees: departure delta-v from v_inf and the low-orbit escape speed
#     (Venus about 3.3 km/s, Mars about 2.1 km/s at Hohmann v_inf); driver
#     energy with a copper sabot that the final station brakes and recovers
#     (regenerative); station momentum restored by ion thrusters on local
#     propellant, less a credit for braking inbound cargo; orbital solar power
#     cost scaled by local sunlight (Venus orbit about 1.9x Earth's, Mars
#     about 0.43x); station capex per kg; container loss rate; heat-shield
#     penalty for faster returns.
#   * Earth-to-destination transport now assumes a mass-driver departure from
#     Earth orbit as well (Earth surface to orbit still needs a launcher):
#     500-50,000 USD/kg, common to both planets.
#
# Changes from v9
#   * RQ3 refocused on high-value pharmaceutical botanicals and cannabis
#     products (PRODUCTS table): paclitaxel via plant cell culture, vinca
#     alkaloids (vincristine class), premium cannabis flower, THC distillate,
#     minor-cannabinoid isolates. Production cost = dry biomass per kg of
#     product x biomass cost + extraction; cannabis carries a compliance cost
#     (US: only FDA-approved and state-licensed medical marijuana moved to
#     Schedule III in April 2026; the rest remains Schedule I) and an optional,
#     speculative "space-grown" price premium.
#   * RQ3 launch: single-stage reusable NTR-H2 to low orbit, mass-driver
#     departure. Two scenarios: rocket (that alone) and full (balloon-lifted
#     launch + winged nuclear ramjet phase heating ambient CO2 to 1.5-3 km/s +
#     rotating-tether catch). Wing/inlet mass is charged to the dry fraction.
#     The fuel-supply premium applies to the reactor hardware.
#   * RQ4 export shuttles: rotating-tether catch on both planets; Venus also
#     gets the balloon launch and nuclear ramjet phase (Mars air too thin).
#   * RQ2: acid-service fluoropolymer collector pads (3-15 kg/m2); in the
#     section scenario the collector hangs on a tether and wind shear between
#     altitudes drives air through it, removing fans when shear exceeds the
#     face velocity; CO2-electrolysis base mass is now stack-only (MOXIE's
#     figure included its compressor), with Mars keeping its compressor factor.
#
# Changes from v8
#   * Greenhouse structure from pressure physics, replacing the assumed
#     hardware masses: membrane mass per m2 = dP * r * SF / (specific strength),
#     plus hydroponic racks and frame. Mars greenhouses may run at low pressure
#     (plants tolerate about 10 kPa; NASA) against 0.6 kPa outside; Venus
#     greenhouses sit near ambient with a small overpressure.
#   * Night storage: each planet uses the lighter of batteries or regenerative
#     fuel cells (NASA RFC target > 550 Wh/kg incl. reactants; reactants made
#     locally on both planets). Venus drift latitude widened to 0-60 deg with
#     the day-night cycle shortening as cos(lat) (Landis 2003: higher latitudes
#     shorten the cycle) at a cos(lat) cost in sunlight.
#   * Kilopower-class fission available to BOTH planets for the night/base
#     load share, chosen per draw only where cheaper than solar + storage.
#     Mass from NASA Fission Surface Power (40 kWe, < 6 t, 10-yr life, HALEU);
#     cost per kWe and a nuclear-fuel supply premium (HALEU availability and
#     regulation) enter capex; replacement follows core life. Venus adds a
#     tether-shield multiplier and the envelope lift penalty. The same fuel
#     supply premium applies to the NTR-H2 export shuttles on both planets.
#   * Mars solar-only colonies must ride out a global dust storm: the array is
#     oversized so storm-time output (5-30% of normal) still covers the
#     critical night/base load; fission removes that need.
#   * Water walls get a dual-use credit on both planets: mass up to the life-
#     support water reserve is not charged twice. Venus shelter areal density
#     widened to 5-40 g/cm2, since RQ1 physics puts Venus SEP dose near zero.
#   * Water prices refreshed from the current RQ2 physics (section draw).
#
# Changes from v7
#   * RQ5 is tested against a materiality margin: D = Venus minus Mars
#     health-adjusted productive hours, minus 5% of scheduled working hours
#     (CFG$rq5_margin_frac). A monotone gravity dose-response guarantees
#     Venus >= Mars, so the margin makes the test ask whether the gap matters.
#     The RQ4 pass-through is unchanged.
#
# Changes from v6
#   * RQ5 (gravity) added as its own one-sided test and passed into RQ4.
#     Health deficit follows a logistic dose-response in gravity whose
#     threshold is sampled from below lunar gravity to 0.6 g (JAXA MHU mice:
#     1/6 g kept soleus mass but only partly kept fiber type, threshold
#     between 1/6 and 1 g). The deficit drives countermeasure exercise time
#     (2.5 h/day at the microgravity limit, ISS practice) and medical
#     attrition / replacement. Crop yield has its own threshold between Moon
#     and Mars gravity (Manzano et al. 2018: mild effects at 0.38 g).
#     RQ5 response: health-adjusted productive hours per colonist-year,
#     Venus minus Mars. RQ4 charges lost hours at a labor value, attrition at
#     a replacement cost, and scales local food by the crop yield factor.
#     Venus at 50-55 km is about 0.89 g; Mars 0.379 g.
#
# Changes from v5
#   * Acid draws by periodic descent into the lower cloud (47.5-50.5 km, peak
#     droplet mass) instead of a fixed collector altitude. Two scenarios for
#     RQ2 and RQ4 (same hypotheses, scenario-conditional results):
#       city    - the whole city descends for short draw campaigns (duty
#                 5-30% of daylight); the habitat must be cooled while low
#       section - a detachable, uncrewed collector section stays low most of
#                 the time (duty 50-90%) on its own buoyant hull
#     Collected acid is stored and processed later at habitat level, so only
#     the collector and fans scale with draw duty. Arrays at draw depth sit
#     under the cloud deck, so fan power carries a reduced-irradiance penalty.
#     Acid buffer tankage scales with the stored mass between draws.
#   * RQ4 hydroponics lighting and CO2 from physics. Daily light integral
#     (DLI) from natural light on each planet is compared with the crop target;
#     the deficit is met by LEDs, whose power and fixture mass enter the colony
#     power budget. Venus: about 4.5x Mars top-of-atmosphere sunlight, diffuse
#     in-cloud light, low-intensity lighting through part of the 50 h night for
#     photoperiod-sensitive crops. Mars: glazed greenhouse behind transparent
#     water walls, mid-latitude sun, dust storms. Crop CO2 is free from Venus
#     air at 0.5-1 atm after acid scrubbing; Mars compresses it from 6 mbar.
#
# Changes from v4
#   * Process equipment mass (formerly "plant_kg_per_kgyr", meaning industrial
#     plant, not crops) is now derived from physics on both planets.
#     Venus: the collector sits at a sampled altitude of 48-55 km. Temperature,
#     pressure and CO2 density come from that altitude (about 27 C and 0.53 atm
#     at 55 km, about 75 C and 1.07 atm at 50 km), so the acid "water ore" is
#     harvested near ambient conditions with no pressure vessels or cryogenics.
#     Cloud mass loading and mass-weighted droplet size follow the altitude
#     (Knollenberg & Hunten 1980 lower vs middle cloud). Mesh-pad capture
#     efficiency follows droplet size against a cut diameter; pad pressure drop
#     is K velocity heads at the face velocity; face area and pad, fan, acid
#     reactor, condenser and CO2-electrolysis masses follow the throughput.
#     Mars: extraction oven mass follows thermal duty; condenser and CO2
#     electrolysis (with the 6 mbar intake compressor) follow throughput.
#   * Both ISRU plants are sized for daylight-only operation (half the hours).
#   * RQ1 solar energetic particles are now event-driven on both planets:
#     Poisson count of large (GLE-class) events per year, heavy-tailed per-event
#     surface dose, and a water-wall storm shelter whose use depends on warning.
#     Mars per-event surface dose is anchored on the Oct 1989 event modeled at
#     Gale (about 3.3 mGy/day, Guo et al. 2018); Venus at 51 km on Herbst et al.
#     2020 (Carrington-class GLE05 about 2.4e-3 mGy/h).
#   * RQ4 adds water-wall shielding capex: Mars walls the whole facility
#     (GCR + SEP), Venus walls only the crew-quarters storm shelter, with an
#     envelope lift penalty. Water is priced from the RQ2 physics.
#
# Changes from v3
#   * Mars water is no longer a lumped kWh/kg guess. It is derived from the
#     ice mass fraction of the mined material, which is tied to site latitude:
#     near-pure excess ice under a 1-2 m ice-cemented cap at about 55-58 deg
#     (Dundas et al. 2018) versus pore ice / hydrated regolith at lower
#     latitudes (Curiosity soil about 1.5-3 wt% H2O, Leshin et al. 2013).
#     Energy = digging ice-cemented ground (with cold-operation penalty) +
#     heating the carried regolith and the ice + sublimation, less heat recovery.
#   * Mars excavation fleet mass scales with material throughput, with
#     abrasion-driven annual spares; Venus collectors carry acid-corrosion spares.
#   * Higher-latitude ice sites now cost Mars sunlight through the shared
#     site latitude (30-60 deg).
#
# Changes from v2
#   * Station-keeping (hold) mode removed: propulsion power about 1.7 MW per
#     colonist made it uneconomic. The colony drifts with the superrotation.
#   * RQ3 runs on reusable NTR with H2 propellant made from acid-derived water;
#     the chemical and CO2-NTR scenarios are dropped.
#   * RQ4 export cost is now derived: reusable NTR-H2 shuttle to low orbit on
#     both planets (rocket equation), plus an orbital launcher fee (Venus mass
#     driver / tether, Mars departure stage).
#   * Fertilizer is primarily recycled black water and inedible biomass on both
#     planets. Venus mineral make-up (P, K, others) comes from robotic surface
#     excavation with balloon ore return to the cloud station, or from Earth if
#     that is cheaper; Venus N is fixed locally and S is an acid byproduct.
#
# Changes from v1 (carried in v2)
#   * Power is derived from physics for both planets (irradiance, cloud
#     transmission, bifacial gain, cell temperature, array mass, storage)
#     instead of a single W/kg guess.
#   * Two Venus operating modes, each run as its own scenario for RQ2 and RQ4:
#       drift - colony rides the superrotating wind (about 50 h day / 50 h night)
#       hold  - colony flies against the wind to stay in daylight, paying the
#               propulsion power 0.5 * rho * v^3 * C_DV * V^(2/3) / eta
#   * Venus water is acid-primary: captured H2SO4 droplets give free water,
#     H2SO4 -> H2O + SO3 dehydration water, and SO3 -> SO2 + 1/2 O2 coproduct
#     oxygen. CO2 electrolysis makes up any O2 shortfall. SO2 is retained as a
#     sulfur / fertilizer / chemical feedstock.
#   * Collector draws from the lower cloud (47.5-50.5 km, Knollenberg & Hunten
#     1980) via a tethered intake or companion aerostat; habitat stays at 50-55 km.
#   * RQ3 adds reusable nuclear thermal rocket shuttles: NTR on ambient CO2
#     (no oxidizer, near-free propellant) and NTR on acid-derived H2.
#   * ISRU plants run in daylight only (oversized by 1 / day fraction); stored
#     energy serves only the life-support share of the colony load.
#   * RQ4 adds habitat thermal control (Mars heating, Venus cooling, heat-pump
#     COP from Carnot fraction), colony power capex and replacement, and
#     fertilizer make-up (Venus: local N and S, imported P, K, others;
#     Mars: local P, K, S and others from regolith, N local or imported).
#
# Statistical design (unchanged from v1)
#   p_MC = (1 + #{D <= 0}) / (n_mc + 1), one-sided, alpha = 0.05 per test, no
#   correction across RQs. H2O ANN surrogate refit on B bootstrap resamples
#   gives per-prediction perturbation sensitivities; paired permuted-predictor
#   refits give per-variable p-values; adonis2 on the local sensitivity matrix.
#   One run per RQ: Venus drifting colony, NTR-H2 shuttles.
#
# Rerun the whole script after any change.
# =============================================================================

suppressPackageStartupMessages({
  library(h2o)
  library(vegan)
})

CFG <- list(
  seed           = 20261013,
  n_mc           = 20000,
  n_train        = 12000,   # RQ4 has about 85 inputs; 12000 rows + 64x32 gave holdout R2 about 0.89
  n_holdout      = 1000,
  n_eval         = 300,
  B              = 100,
  B_null         = 40,      # smallest attainable p = 1/41 = 0.024
  alpha          = 0.05,
  rq5_margin_frac = 0.05,   # RQ5 materiality margin: share of scheduled working hours
  hidden         = c(64, 32),
  cat_encoding   = "OneHotInternal",        # with use_all_factor_levels = TRUE; Eigen fit RQ3 worse (0.81 vs 0.99)
  activation     = "RectifierWithDropout",
  hidden_dropout = c(0.05, 0.05),           # one ratio per hidden layer; 0.05 benchmarked best (H2O default 0.5 cut R2 by ~0.1-0.16)
  epochs         = 40,
  l2             = 1e-5,
  perturb_sd     = 0.10,
  permanova_perm = 999,
  infeasible_cost_usd_kg = 1e9,
  nthreads       = max(1, parallel::detectCores() - 2),
  max_mem        = "16G",
  out_dir        = if (dir.exists("C:/mars_venus_paper")) "C:/mars_venus_paper/output/v26"
                   else "venus_mars_mc_ann_v26_output",
  runs           = list(c("RQ1", "none"), c("RQ2", "city"), c("RQ2", "section"),
                        c("RQ3", "rocket"), c("RQ3", "full"), c("RQ3", "protein"), c("RQ3", "isotope"), c("RQ4", "city"), c("RQ4", "section"),
                        c("RQ5", "none"))
)

if (Sys.getenv("VM_QUICK") == "1") {
  CFG$n_mc <- 3000; CFG$n_train <- 1200; CFG$n_holdout <- 400; CFG$n_eval <- 80
  CFG$B <- 4; CFG$B_null <- 3; CFG$epochs <- 10; CFG$permanova_perm <- 199
  CFG$max_mem <- "4G"
  CFG$out_dir <- if (dir.exists("C:/mars_venus_paper")) "C:/mars_venus_paper/output/v26_quicktest"
                 else "venus_mars_mc_ann_v26_quicktest"
}

if (1 / (CFG$B_null + 1) >= CFG$alpha)
  warning(sprintf("B_null = %d: smallest attainable null-arm p is %.3f, not below alpha. Use B_null >= %d.",
                  CFG$B_null, 1 / (CFG$B_null + 1), ceiling(1 / CFG$alpha)))
dir.create(CFG$out_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(CFG$seed)

# -----------------------------------------------------------------------------
# PARAMETER TABLE
# mode: both | drift | hold (Venus operating mode the row applies to)
# Basis tags in 'source': [S] sourced, [D] derived, [A] working assumption,
# [VERIFY] to be replaced with a sourced value before submission.
# -----------------------------------------------------------------------------
p <- function(name, rq, dist, a, b, units, source, mode = "both") {
  data.frame(name = name, rq = rq, dist = dist, a = a, b = b, units = units,
             source = source, mode = mode, stringsAsFactors = FALSE)
}

# Shared power block, attached to RQ2 and RQ4
power_block <- function(rq) rbind(
  p("cell_eff", rq, "unif", 0.20, 0.32, "-", "[A] Thin-film to multijunction cell efficiency at 25 C"),
  p("cell_temp_coef", rq, "unif", -0.0045, -0.0025, "1/C", "[S] Typical Si to III-V temperature coefficients"),
  p("array_kg_m2", rq, "unif", 0.3, 1.5, "kg/m2", "[A] Flexible array areal mass incl. structure"),
  p("pc_eff", rq, "unif", 0.90, 0.95, "-", "[S] Power conditioning efficiency (Landis & Colozza use 0.95)"),
  p("battery_Wh_kg", rq, "unif", 150, 300, "Wh/kg", "[S] Li-ion pack level"),
  p("rfc_Wh_kg", rq, "unif", 400, 1000, "Wh/kg", "[S/D] NASA RFC target > 550 Wh/kg incl. reactants; local H2/O2 leaves tanks + stack (about 850 Wh/kg at 6 wt% H2 tankage)"),
  p("rfc_stack_kg_per_kW", rq, "unif", 5, 20, "kg/kW", "[A] RFC stack + balance of plant"),
  p("dod", rq, "unif", 0.7, 0.9, "-", "[A] Usable depth of discharge"),
  p("ls_frac", rq, "unif", 0.2, 0.5, "-", "[A] Share of colony load that must run through the night"),
  p("life_yr", rq, "unif", 10, 20, "yr", "[A] Equipment service life"),
  p("transport_base_usd_kg", rq, "logunif", 500, 5e4, "USD/kg", "[A] Earth launch + mass-driver departure from Earth orbit, delivered; common random number [VERIFY]"),
  p("mars_transport_mult", rq, "unif", 0.9, 1.3, "-", "[A] Propulsive landing penalty"),
  p("venus_transport_mult", rq, "unif", 0.8, 1.3, "-", "[A] Entry + aerostat deployment"),
  # Mars insolation chain
  p("mars_toa_Wm2", rq, "unif", 493, 717, "W/m2", "[D] 1361 W/m2 at 1.666 to 1.381 AU (aphelion to perihelion)"),
  p("mars_tau_trans", rq, "unif", 0.55, 0.85, "-", "[A] Clear-season atmospheric transmission (optical depth about 0.3-0.6) [VERIFY]"),
  p("mars_storm_factor", rq, "unif", 0.85, 1.0, "-", "[A] Annualized dust-storm loss [VERIFY]"),
  p("mars_storm_output", rq, "unif", 0.05, 0.30, "-", "[S/A] Array output during a global dust storm (Opportunity 2018: optical depth about 10.8)"),
  p("mars_panel_dust", rq, "unif", 0.75, 1.0, "-", "[A] Panel dust accumulation with cleaning"),
  p("mars_site_lat", rq, "unif", 30, 60, "deg", "[S] SWIM mid-latitude ice; near-pure scarp ice at 55-58 deg (Dundas et al. 2018)"),
  p("mars_cell_C", rq, "unif", -40, 0, "C", "[A] Daytime cell temperature"),
  p("mars_night_h", rq, "unif", 11.5, 13.5, "h", "[D] Sol 24.66 h, season and latitude"),
  # Venus insolation chain
  p("venus_alt_km", rq, "unif", 54, 60, "km", "[S/D] Habitat float altitude: above or at the top of the convective layer, below the upper-cloud shear (>~62 km); VIRA 0.62-0.23 atm"),
  p("venus_cloud_trans", rq, "unif", 0.3, 0.7, "-", "[D/A] Share of 2614 W/m2 reaching 50-55 km; 0.2-0.5 at the 45 km cloud base (Landis & Colozza)"),
  p("bifacial_gain", rq, "unif", 1.3, 1.9, "-", "[S] Downward-facing cells nearly match sunward output (Landis 2003)"),
  p("drift_night_h", rq, "unif", 40, 55, "h", "[S] About 50 h day / 50 h night riding a 4-day circulation (Landis 2003)"),
  p("disrupt_light_loss", rq, "unif", 0, 0.05, "-", "[A] Sunlight lost to disruption-front passages when the habitat floats below 56.5 km"),
  p("drift_lat", rq, "unif", 0, 50, "deg", "[S/D] Drift latitude; cycle shortens as cos(lat) assuming latitude-independent zonal wind [VERIFY] (Landis 2003)")
)


# Mass-driver departure block (both planets), attached to RQ3 and RQ4
depart_block <- function(rq) rbind(
  p("driver_eff", rq, "unif", 0.6, 0.9, "-", "[A] Coilgun electrical-to-kinetic efficiency"),
  p("sabot_frac", rq, "unif", 0.2, 0.5, "kg/kg", "[A] Copper sabot mass per kg container"),
  p("sabot_regen", rq, "unif", 0.5, 0.9, "-", "[A] Sabot kinetic energy recovered when the final station brakes it"),
  p("ion_isp_s", rq, "unif", 2000, 5000, "s", "[S] Gridded ion / Hall thruster range"),
  p("ion_eff", rq, "unif", 0.5, 0.7, "-", "[S] Thruster efficiency"),
  p("inbound_credit", rq, "unif", 0, 0.8, "-", "[A] Share of station momentum restored by braking inbound cargo"),
  p("station_ref_usd_kg", rq, "logunif", 50, 2000, "USD/kg", "[A] Driver constellation capital per kg at a reference 1,000 t/yr; scaled by throughput"),
  p("traffic_kg_colonist_yr", rq, "logunif", 500, 5000, "kg/colonist/yr", "[A] Two-way orbital traffic (imports + exports) per colonist through tethers and drivers"),
  p("tether_mass_ratio", rq, "unif", 60, 200, "kg/kg payload", "[S/D] HASTOL Phase I facility about 160-200 x a 14 t payload; Phase II tether about 3 x lighter"),
  p("tether_unit_mult", rq, "unif", 0.3, 1.2, "-", "[A] Facility cost per kg as a multiple of the planet's transport cost (local ballast cheaper than imported fiber)"),
  p("catches_per_yr", rq, "unif", 20, 100, "1/yr", "[A] Catches per tether per year; HASTOL restores orbit over 'a number of days'"),
  p("capital_charge", rq, "unif", 0.08, 0.20, "1/yr", "[A] Annual capital charge plus O&M on tether facilities"),
  p("scale_exp", rq, "unif", 0.6, 0.9, "-", "[S/A] Cost-capacity exponent (six-tenths rule; process plants typically 0.6-0.8)"),
  p("solar_usd_kWh_1AU", rq, "logunif", 0.5, 20, "USD/kWh", "[A] Orbital solar electricity at 1 AU; scaled by local sunlight"),
  p("container_loss", rq, "unif", 0, 0.03, "-", "[A] Containers lost to targeting errors"),
  p("entry_speed_penalty", rq, "unif", 0.02, 0.05, "per km/s", "[A] Payload-fraction loss per km/s of Earth entry speed above 11.5 km/s"),
  p("imported_frac_dry", rq, "unif", 0.05, 0.20, "-", "[A] Imported avionics / precision share of shuttle dry mass"),
  p("reactor_frac_dry", rq, "unif", 0.15, 0.35, "-", "[A] NTR reactor share of shuttle dry mass"),
  p("local_mfg_usd_kg", rq, "logunif", 100, 3000, "USD/kg", "[A] Local carbon-composite airframe fabrication"),
  p("reactor_life_flights", rq, "logunif", 20, 200, "flights", "[S/A] NRX-A6 62 min continuous run (1967); minutes of burn per ascent")
)

# USD per kg cargo for mass-driver departure from low orbit (derived)
departure_cost <- function(d, mu, r_orbit_m, v_inf_km_s, sun_ratio) {
  v_circ <- sqrt(mu / r_orbit_m); v_esc <- sqrt(2 * mu / r_orbit_m)
  dv <- sqrt((v_inf_km_s * 1000)^2 + v_esc^2) - v_circ                  # m/s
  e_driver <- 0.5 * dv^2 * (1 + d$sabot_frac) / d$driver_eff -
              0.5 * dv^2 * d$sabot_frac * d$sabot_regen                    # J/kg cargo
  ve <- d$ion_isp_s * 9.80665
  e_ion <- dv * (1 - d$inbound_credit) * ve / (2 * d$ion_eff)            # J/kg cargo
  usd_kWh <- d$solar_usd_kWh_1AU / sun_ratio
  ((e_driver + e_ion) / 3.6e6 * usd_kWh + d$station_capex_usd_kg) / (1 - d$container_loss)
}
MU_VENUS <- 3.2486e14; R_LVO <- (6051.8 + 250) * 1000
MU_MARS  <- 4.2828e13; R_LMO <- (3389.5 + 250) * 1000
SUN_VENUS <- 2614 / 1361; SUN_MARS <- 586 / 1361

# Gravity / health block, attached to RQ5 and RQ4
gravity_block <- function(rq) rbind(
  p("g_threshold", rq, "unif", 0.10, 0.60, "g", "[S/A] Health dose-response midpoint; JAXA MHU mice place it between 1/6 g and 1 g for fiber type"),
  p("g_width", rq, "unif", 0.03, 0.10, "g", "[A] Steepness of the health dose-response"),
  p("cm_hours_max", rq, "unif", 2.0, 2.5, "h/day", "[S] ISS countermeasure exercise at the microgravity limit"),
  p("attrition_max", rq, "unif", 0.02, 0.15, "1/yr", "[A] Share of colonist-years lost to fractures, illness, evacuation at the microgravity limit"),
  p("work_hours_yr", rq, "unif", 2000, 3000, "h/yr", "[A] Scheduled productive hours per colonist"),
  p("crop_g_threshold", rq, "unif", 0.10, 0.38, "g", "[S] Root meristem threshold between Moon and Mars gravity (Manzano et al. 2018)"),
  p("crop_def_max", rq, "unif", 0, 0.15, "-", "[A] Maximum crop yield loss well below threshold; mild at 0.38 g")
)

health_model <- function(d) {
  gV <- 0.889; gM <- 0.379
  def <- function(g) 1 / (1 + exp((g - d$g_threshold) / d$g_width))
  cdef <- function(g) 1 / (1 + exp((g - d$crop_g_threshold) / 0.05))
  hrs <- function(g) (d$work_hours_yr - d$cm_hours_max * def(g) * 365) * (1 - d$attrition_max * def(g))
  list(defV = def(gV), defM = def(gM), hrsV = hrs(gV), hrsM = hrs(gM),
       lostV = d$work_hours_yr - hrs(gV), lostM = d$work_hours_yr - hrs(gM),
       attrV = d$attrition_max * def(gV), attrM = d$attrition_max * def(gM),
       cropV = 1 - d$crop_def_max * cdef(gV), cropM = 1 - d$crop_def_max * cdef(gM))
}

PARAMS <- rbind(
  # ---------------- RQ1: annual dose equivalent (unchanged) ----------------
  p("mars_gcr_mGy_day", "RQ1", "unif", 0.17, 0.25, "mGy/day", "[S] Hassler et al. 2014: 0.21 +/- 0.04 mGy/day"),
  p("mars_Q", "RQ1", "unif", 2.8, 3.3, "-", "[S] Hassler et al. 2014: quality factor 3.05 +/- 0.26"),
  p("sep_rate_yr", "RQ1", "unif", 0.5, 1.5, "events/yr", "[S] About 73 ground-level-enhancement events since 1942"),
  p("sep_count_u", "RQ1", "unif", 0, 1, "-", "[D] Quantile of the Poisson event count for this year (count = qpois(u, rate))"),
  p("sep_scatter", "RQ1", "unif", -1, 1, "log10", "[A] Year's event-severity scatter about the typical event, x0.1 to x10"),
  p("mars_sep_event_mGy", "RQ1", "logunif", 1, 300, "mGy/event", "[S/A] Oct 1989 at Gale about 3.3 mGy/day over several days; extreme hard-spectrum events far higher [VERIFY tail]"),
  p("sep_Q", "RQ1", "unif", 1.5, 2.5, "-", "[A] Quality factor for SEP proton spectra behind atmosphere"),
  p("shelter_transmission", "RQ1", "unif", 0.1, 0.4, "-", "[A] Dose transmission of a 20-40 g/cm2 water-wall shelter for SEP spectra [VERIFY]"),
  p("shelter_warning", "RQ1", "unif", 0.6, 0.95, "-", "[A] Share of event dose received after crew reaches shelter (warning lead time)"),
  p("mars_hab_transmission", "RQ1", "unif", 0.3, 0.8, "-", "[VERIFY] GCR transmission of habitat walls (regolith berm or water walls)"),
  p("mars_time_inside", "RQ1", "unif", 0.80, 0.95, "-", "[A] Time inside shielded habitat"),
  p("venus_alt_km", "RQ1", "unif", 54, 60, "km", "[S/D] Habitat float altitude (v24 weather band)"),
  p("venus_gcr51_mGy_h", "RQ1", "unif", 4.5e-5, 6.9e-5, "mGy/h", "[S] Herbst et al. 2020, 51 km"),
  p("venus_dose_growth_per_km", "RQ1", "unif", 1.45, 1.55, "x per km", "[D] Herbst 2020, 51 to 62 km log-linear"),
  p("venus_Q", "RQ1", "unif", 1.5, 3.0, "-", "[VERIFY] Deep-atmosphere quality factor"),
  p("venus_sep_event_mGy", "RQ1", "logunif", 1e-3, 0.3, "mGy/event", "[S] Herbst 2020: 51 km GLE doses 4.5e-5 to 9e-3 mGy/h, up to about a day; higher toward 55 km"),

  # ---------------- RQ2: delivered cost of H2O + O2 basket ----------------
  power_block("RQ2"),
  p("basket_h2o_frac", "RQ2", "unif", 0.3, 0.7, "-", "[A] H2O share of the product basket"),
  p("mars_ice_u", "RQ2", "unif", 0, 1, "-", "[D] Position within the latitude-dependent ice-fraction range (see mars_ice_frac())"),
  p("mars_strip_ratio", "RQ2", "unif", 0, 1, "kg/kg", "[A] Overburden (ice-cemented cap) moved per kg of ice-bearing material"),
  p("mars_dig_kWh_kg", "RQ2", "logunif", 0.003, 0.05, "kWh/kg", "[A] Ripping/excavating ice-cemented ground and hauling [VERIFY]"),
  p("mars_cold_penalty", "RQ2", "unif", 0.1, 0.4, "-", "[A] Extra power to keep machinery operable at about -63 C"),
  p("mars_dT_K", "RQ2", "unif", 60, 100, "K", "[D] Heating from about 200-215 K to extraction temperature"),
  p("mars_heat_recovery", "RQ2", "unif", 0.0, 0.6, "-", "[A] Recuperation of regolith heat and latent heat"),
  p("mars_extract_eff", "RQ2", "unif", 0.5, 0.8, "-", "[A] Thermal extraction efficiency"),
  p("mars_excavator_tput", "RQ2", "logunif", 100, 5000, "kg/yr per kg machine", "[A] Upper bound from RASSOR-class bucket drums in loose regolith; ice-cemented ground is far harder [VERIFY]"),
  # ---- Mars process equipment (physics-derived) ----
  p("oven_kg_per_kW", "RQ2", "unif", 2, 10, "kg/kW_th", "[A] Extraction oven / heater specific mass per kW of thermal duty [VERIFY]"),
  p("mars_compressor_factor", "RQ2", "unif", 1.2, 1.6, "-", "[A] CO2 electrolysis mass multiplier for compressing 6 mbar intake"),
  p("mars_process_spares", "RQ2", "unif", 0.02, 0.15, "1/yr", "[A] Process equipment spares (same range as Venus acid service)"),
  # ---- shared process equipment ----
  p("condenser_kg_per_kgh", "RQ2", "unif", 20, 100, "kg per kg/h", "[A] Condenser / purifier per kg/h water"),
  p("soxe_kg_per_kgh", "RQ2", "unif", 200, 1200, "kg per kg/h O2", "[D] Stack-only: MOXIE 17.1 kg at 0.01 kg/h (1700 incl. compressor) / 1.2-1.6; Mars adds its compressor factor"),
  p("bop_mult", "RQ2", "unif", 1.2, 1.6, "-", "[A] Balance of plant: piping, tanks, controls"),
  p("mars_spares_frac", "RQ2", "unif", 0.1, 0.5, "1/yr", "[A] Abrasive dust wear: spares as share of fleet mass per year"),
  p("mars_purify_kWh_kg", "RQ2", "unif", 0.02, 0.3, "kWh/kg", "[A] Perchlorate / brine removal"),
  p("mars_o2_kWh_kg", "RQ2", "unif", 12, 20, "kWh/kg", "[S] MOXIE scale-up 12.5-15 kWh/kg + CO2 compression"),
  p("fan_eff", "RQ2", "unif", 0.5, 0.7, "-", "[A] Fan efficiency"),
  p("venus_acid_wt", "RQ2", "unif", 0.75, 0.96, "-", "[S] H2SO4 mass fraction of droplets"),
  p("evap_kWh_kg", "RQ2", "unif", 0.8, 1.5, "kWh/kg", "[D] Boiling free water off concentrated acid"),
  p("dehyd_kWh_kg_acid", "RQ2", "unif", 0.6, 1.5, "kWh/kg H2SO4", "[D] H2SO4(l) -> H2O + SO3, +176 kJ/mol (0.5 kWh/kg) + heating, recuperated"),
  p("so3_decomp_kWh_kg_O2", "RQ2", "unif", 2.0, 4.0, "kWh/kg O2", "[D] SO3 -> SO2 + 1/2 O2, +99 kJ/mol (1.72 kWh/kg O2) at about 850 C, recuperated"),
  p("venus_o2_kWh_kg", "RQ2", "unif", 7, 14, "kWh/kg", "[A] CO2 electrolysis at about 1 bar for O2 shortfall"),
  # ---- Venus collector and process equipment (physics-derived) ----
  p("collector_alt_km", "RQ2", "unif", 47.5, 50.5, "km", "[S] Draw depth in the lower cloud, peak droplet mass (K&H 1980 lower cloud 47.5-50.5 km)"),
  p("draw_duty", "RQ2", "unif", 0.05, 0.30, "-", "[A] Share of daylight hours the whole city spends at draw depth", "city"),
  p("draw_duty", "RQ2", "unif", 0.50, 0.90, "-", "[A] Share of daylight hours the collector section spends at draw depth", "section"),
  p("section_hull_mult", "RQ2", "unif", 1.3, 2.0, "-", "[A] Buoyant hull + structure multiplier on collector and fan mass", "section"),
  p("depth_irr_factor", "RQ2", "unif", 0.4, 0.8, "-", "[D/A] Array output at draw depth vs habitat level (more cloud overhead; 0.2-0.5 of TOA at 45 km)"),
  p("draws_per_yr", "RQ2", "unif", 12, 52, "1/yr", "[A] Draw cycles (descent and return) per year"),
  p("tank_kg_per_kg", "RQ2", "unif", 0.03, 0.10, "kg/kg", "[A] Acid-resistant buffer tankage per kg stored"),
  p("aerosol_spread", "RQ2", "unif", -0.4, 0.4, "log10", "[D] Scatter about the altitude-dependent mean loading (K&H 1980: 10-20 mg/m3 average, 100-200 peaks)"),
  p("droplet_size_factor", "RQ2", "unif", 0.7, 1.3, "-", "[S] Mode 2 about 2.1-2.9 um; mode 3 about 7 um in lower cloud (K&H 1980)"),
  p("face_v_m_s", "RQ2", "unif", 1.5, 3.5, "m/s", "[S] Mesh pad design velocity about 3 m/s (10 ft/s) in Earth service, density-adjusted"),
  p("pad_K", "RQ2", "unif", 20, 100, "velocity heads", "[S] Dry mesh pad drop of order 100-500 Pa ('a few inches of water' max)"),
  p("pad_d50_um", "RQ2", "unif", 1.5, 5.0, "um", "[S] Cut diameter: co-knit pads near 2 um (99.9% at >= 2 um claimed) to plain 4-inch mesh (72% at 5 um)"),
  p("pad_sharpness", "RQ2", "unif", 2, 4, "-", "[A] Steepness of the grade-efficiency curve"),
  p("collector_kg_m2", "RQ2", "unif", 3, 15, "kg/m2", "[D] Acid-service PTFE/PFA or glass-fiber pads (about 1/3 the density of steel mesh, which corrodes in H2SO4) + frame"),
  p("shear_wind_m_s", "RQ2", "unif", 2, 20, "m/s", "[S/A] Relative wind at a collector tethered below the city; lower-cloud winds vary 60 m/s to 80 m/s (Akatsuki 2016 equatorial jet) [VERIFY]", "section"),
  p("tether_hull_mult", "RQ2", "unif", 1.1, 1.5, "-", "[A] Tether, drag-control and gust-load margin for a collector section crossing the convective layer", "section"),
  p("disrupt_collect_loss", "RQ2", "unif", 0, 0.10, "-", "[A] Collection time lost to passages of the 47.5-56.5 km planetary-scale cloud disruption (Peralta et al. 2020)"),
  p("fan_W_per_kg", "RQ2", "unif", 500, 2000, "W/kg", "[A] Fan + motor specific power"),
  p("reactor_kg_per_kgh", "RQ2", "unif", 5, 50, "kg per kg/h", "[A] Acid evaporator, dehydrator and 850 C SO3 splitter per kg/h of solution [VERIFY]"),
  p("venus_spares_frac", "RQ2", "unif", 0.02, 0.15, "1/yr", "[A] Acid-service spares; Earth sulfuric acid plants run a few % of capital per year with PTFE/glass/alloy linings [VERIFY]"),

  # ---------------- RQ3: high-value botanical pharmaceutical exports ----------------
  p("product_price_usd_kg", "RQ3", "product", NA, NA, "USD/kg", "See PRODUCTS table", "export"),
  p("biomass_u", "RQ3", "unif", 0, 1, "-", "[D] Position in the product's biomass-per-kg range (log scale)", "export"),
  p("biomass_cost_usd_kg", "RQ3", "logunif", 5, 200, "USD/kg dry", "[A] Venus hydroponic / cell-culture biomass cost", "export"),
  p("extraction_usd_kg", "RQ3", "logunif", 100, 2000, "USD/kg", "[A] Extraction and purification per kg product (not flower)", "export"),
  p("cannabis_compliance_frac", "RQ3", "unif", 0.05, 0.30, "-", "[A] Licensing / compliance cost share of price; US medical channel Schedule III since Apr 2026", "export"),
  p("space_premium", "RQ3", "unif", 1.0, 1.5, "x", "[A, speculative] 'Space-grown' price premium, cannabis classes only", "export"),
  p("ascent_dv_km_s", "RQ3", "unif", 8.0, 10.5, "km/s", "[D] Cloud level to LVO: 7.3 km/s orbital + losses"),
  p("isp_s", "RQ3", "unif", 850, 950, "s", "[S] Reusable NTR on H2 (NERVA / DRACO class)"),
  p("dry_frac", "RQ3", "unif", 0.15, 0.25, "-", "[A] NTR + low-density LH2 tankage"),
  p("prop_kWh_kg", "RQ3", "unif", 150, 450, "kWh/kg", "[D] About 9 kg acid-derived H2O per kg H2 + electrolysis + liquefaction"),
  p("rockoon_save_km_s", "RQ3", "unif", 0.2, 0.6, "km/s", "[A] Drag and gravity loss saved by balloon-lifting to about 70 km", "full"),
  p("ramjet_v_km_s", "RQ3", "unif", 1.5, 3.0, "km/s", "[A] Nuclear ramjet on ambient CO2 (Project Pluto principle); reactor temperature limits top speed", "full"),
  p("wing_frac", "RQ3", "unif", 0.03, 0.10, "-", "[A] Wings, inlet and heat shield as added dry fraction", "full"),
  p("tether_tip_km_s", "RQ3", "unif", 2.5, 3.5, "km/s", "[S] Rotating-tether tip speed (HASTOL study 3.5 km/s); momentum restored by catching inbound cargo", "full"),
  p("venus_v_inf_km_s", "RQ3", "unif", 2.5, 5.0, "km/s", "[D] Hohmann about 2.5 km/s to faster returns"),
  # ---- protein scenario ----
  p("rockoon_save_km_s", "RQ3", "unif", 0.2, 0.6, "km/s", "[A] Balloon-lifted launch loss saving", "protein"),
  p("ramjet_v_km_s", "RQ3", "unif", 1.5, 3.0, "km/s", "[A] Nuclear ramjet on CO2", "protein"),
  p("wing_frac", "RQ3", "unif", 0.03, 0.10, "-", "[A] Wing/inlet dry fraction", "protein"),
  p("tether_tip_km_s", "RQ3", "unif", 2.5, 3.5, "km/s", "[S] Rotating-tether tip speed (HASTOL 3.5 km/s)", "protein"),
  p("market_u", "RQ3", "unif", 0, 1, "-", "[A] Market: < 0.5 low Earth orbit / cislunar, >= 0.5 Mars orbit", "protein"),
  p("protein_kWh_kg", "RQ3", "unif", 18, 85, "kWh/kg protein", "[S] 17.8 thermodynamic floor (Wise 2022) to 65-85 (Smaje 2024); Factory 01 about 62.5", "protein"),
  p("protein_frac", "RQ3", "unif", 0.60, 0.70, "-", "[S] Protein share of dry microbial biomass (Solein about 65-70%)", "protein"),
  p("ferm_kg_per_kgyr", "RQ3", "logunif", 0.05, 0.5, "kg per kg/yr", "[A] Bioreactor plant mass per kg/yr capacity [VERIFY]", "protein"),
  p("N_kWh_kg", "RQ3", "unif", 10, 15, "kWh/kg N", "[A] Haber-Bosch with electrolytic H2 (N2 is 3.5% of Venus air)", "protein"),
  p("venus_v_inf_mars", "RQ3", "unif", 5.5, 6.5, "km/s", "[D] Venus-to-Mars Hohmann departure v_inf about 5.8 km/s", "protein"),
  p("earth_v_inf_mars", "RQ3", "unif", 2.9, 3.5, "km/s", "[D] Earth-to-Mars Hohmann departure v_inf about 2.9 km/s", "protein"),
  p("earth_launch_usd_kg", "RQ3", "logunif", 100, 7000, "USD/kg", "[S/A] SpaceX rideshare $7,000/kg (Feb 2026) down to Starship-class target about $100/kg", "protein"),
  p("earth_protein_usd_kg", "RQ3", "unif", 5, 25, "USD/kg", "[A] Earth production cost of protein product", "protein"),
  # ---- isotope co-product scenario (He-3 from collector air, D2O from acid-derived water) ----
  p("collector_alt_km", "RQ3", "unif", 47.5, 50.5, "km", "[S] Draw depth in the lower cloud (K&H 1980 lower cloud 47.5-50.5 km)", "isotope"),
  p("aerosol_spread", "RQ3", "unif", -0.4, 0.4, "log10", "[D] Scatter about the altitude-dependent mean cloud loading (K&H 1980)", "isotope"),
  p("droplet_size_factor", "RQ3", "unif", 0.7, 1.3, "-", "[S] Droplet mode sizes (K&H 1980)", "isotope"),
  p("pad_d50_um", "RQ3", "unif", 1.5, 5.0, "um", "[S] Mesh pad cut diameter", "isotope"),
  p("pad_sharpness", "RQ3", "unif", 2, 4, "-", "[A] Grade-efficiency steepness", "isotope"),
  p("venus_acid_wt", "RQ3", "unif", 0.75, 0.96, "-", "[S] H2SO4 mass fraction of droplets", "isotope"),
  p("venus_He_ppm", "RQ3", "logunif", 4, 36, "ppm (molar)", "[S] Pioneer Venus about 12 (+24/-8) ppm He", "isotope"),
  p("he3_ratio", "RQ3", "logunif", 1.4e-6, 3e-4, "3He/4He", "[S/A] Unmeasured on Venus: Earth air 1.4e-6 to the Pioneer Venus upper limit 3e-4 (Hoffman et al. 1980) [VERIFY by probe]", "isotope"),
  p("he_recovery", "RQ3", "unif", 0.3, 0.9, "-", "[A] Share of collector-air He captured", "isotope"),
  p("he_kWh_kg_air", "RQ3", "logunif", 2e-4, 5e-2, "kWh/kg air", "[A] He separation from collector exhaust; thermodynamic minimum about 3e-6, membrane compression of the whole stream about 0.05 [VERIFY]", "isotope"),
  p("he4_kWh_kg", "RQ3", "unif", 5, 20, "kWh/kg He", "[A] Cryogenic 3He/4He separation per kg of bulk He (liquefaction class)", "isotope"),
  p("he_capex_mult", "RQ3", "unif", 0.2, 1.0, "-", "[A] He plant capital and upkeep as a multiple of its energy cost", "isotope"),
  p("he3_price_usd_kg", "RQ3", "logunif", 5e6, 1.11e8, "USD/kg", "[S/A] $2,000-15,000 per liter STP (Quantum Insider 2025) = $15M-111M/kg; floor $5M allows for lunar/Venus supply response", "isotope"),
  p("venus_DH", "RQ3", "unif", 0.0156, 0.025, "D/H", "[S] Venus water about 100-150 x VSMOW (1.56e-4)", "isotope"),
  p("d2o_recovery", "RQ3", "unif", 0.5, 0.9, "-", "[A] Share of feed deuterium recovered as D2O", "isotope"),
  p("d2o_ref_kWh_kg", "RQ3", "logunif", 2000, 33000, "kWh/kg D2O", "[D] Natural-feed reference: Girdler-sulfide > 10 t steam/kg; electrolytic dissociation about 120 GJ/kg (Miller); scaled by feed D content", "isotope"),
  p("d2o_synergy", "RQ3", "unif", 0, 0.7, "-", "[A] Share of enrichment work carried by electrolysis already run for NTR H2", "isotope"),
  p("d2o_capex_mult", "RQ3", "unif", 0.3, 1.0, "-", "[A] Enrichment plant capital and upkeep as a multiple of its energy cost", "isotope"),
  p("d2o_price_usd_kg", "RQ3", "logunif", 270, 1000, "USD/kg", "[S] US DOE paid $8.6M for 32 t (2016), about $270/kg; Miller about $300/kg; upper end reagent / fusion grade [A]", "isotope"),
  p("earth_launch_usd_kg", "RQ3", "logunif", 100, 7000, "USD/kg", "[S/A] Earth launch to LEO, rideshare $7,000/kg to Starship-class $100/kg", "isotope"),
  p("rockoon_save_km_s", "RQ3", "unif", 0.2, 0.6, "km/s", "[A] Balloon-lifted launch loss saving", "isotope"),
  p("ramjet_v_km_s", "RQ3", "unif", 1.5, 3.0, "km/s", "[A] Nuclear ramjet on CO2", "isotope"),
  p("wing_frac", "RQ3", "unif", 0.03, 0.10, "-", "[A] Wing/inlet dry fraction", "isotope"),
  p("tether_tip_km_s", "RQ3", "unif", 2.5, 3.5, "km/s", "[S] Rotating-tether tip speed (HASTOL 3.5 km/s)", "isotope"),
  p("downmass_usd_kg", "RQ3", "logunif", 50, 10000, "USD/kg", "[S/A] Earth orbit to ground, consolidated (He-3); Dragon CRS about $63k/kg today, mature return market $5-10k/kg projected, reusable-shuttle backhaul lower [VERIFY]", "isotope"),
  p("fuel_supply_premium", "RQ3", "unif", 0, 0.5, "-", "[A] HALEU availability and regulatory premium on reactor hardware"),
  p("colony_size_n", "RQ3", "logunif", 100, 10000, "colonists", "[A] Colony population; sets orbital traffic"),
  p("container_LD", "RQ3", "unif", 1, 4, "-", "[A] Container/sabot length-to-diameter ratio (author: longer containers); slender bodies above about 3 need aerocapture stability control [VERIFY]"),
  p("aero_shell_c", "RQ3", "unif", 0.2, 0.6, "kg/kg", "[A] Aeroshell + structure mass per kg product at L/D = 1 (aerocapture to orbit, lighter than full entry)"),
  p("hardware_usd_kg", "RQ3", "logunif", 5e3, 5e5, "USD/kg", "[A] Imported hardware (avionics, reactor) incl. transport"),
  p("vehicle_reuse", "RQ3", "unif", 100, 300, "flights", "[A] Airframe reuse with local maintenance (author's target about 200)"),
  p("downmass_usd_kg", "RQ3", "logunif", 50, 10000, "USD/kg", "[S/A] Earth orbit to ground, consolidated; Dragon CRS about $63k/kg today, mature return market $5-10k/kg projected, reusable-shuttle backhaul lower [VERIFY]", "export"),

  # ---------------- RQ4: colony NPV, Venus minus Mars ----------------
  power_block("RQ4"),
  p("colony_load_kW", "RQ4", "unif", 20, 100, "kW/colonist", "[A] Average electric load per colonist"),
  p("food_import_kg_yr", "RQ4", "unif", 500, 800, "kg/colonist/yr", "[S] About 1.8 kg/day packaged food"),
  p("venus_subst_frac", "RQ4", "unif", 0.5, 0.95, "-", "[A] Local food share"),
  p("mars_subst_frac", "RQ4", "unif", 0.3, 0.8, "-", "[A] Local food share"),
  # greenhouse structure from pressure physics
  p("hydro_kg_m2", "RQ4", "unif", 10, 30, "kg/m2 crop", "[A] Hydroponic racks, pumps, nutrient loop (both planets)"),
  p("env_area_factor", "RQ4", "unif", 1.5, 3.0, "m2/m2", "[A] Envelope area per m2 of growing area (incl. stacking)"),
  p("mars_gh_kPa", "RQ4", "unif", 10, 101, "kPa", "[S] Plants tolerate about 10 kPa (NASA low-pressure greenhouse work) up to 1 atm"),
  p("venus_gh_dp_kPa", "RQ4", "unif", 0.5, 5, "kPa", "[A] Small overpressure to keep acid aerosol out"),
  p("gh_radius_m", "RQ4", "unif", 3, 10, "m", "[A] Envelope radius of curvature"),
  p("gh_SF", "RQ4", "unif", 3, 4, "-", "[S] NASA inflatable structure safety factor about 4"),
  p("spec_strength_kJ_kg", "RQ4", "logunif", 20, 300, "kN*m/kg", "[A] Transparent-film (ETFE-class) to fiber-restrained envelope"),
  p("frame_mult", "RQ4", "unif", 1.2, 1.6, "-", "[A] Frame, seams, airlocks"),
  # Kilopower-class fission (both planets)
  p("fission_kg_per_kWe", "RQ4", "unif", 120, 250, "kg/kWe", "[S] NASA Fission Surface Power: 40 kWe, mass goal < 6 t (>= 150 kg/kWe)"),
  p("fission_usd_per_kWe", "RQ4", "logunif", 2e5, 5e6, "USD/kWe", "[A] Hardware + fuel + licensing; no FSP program total published [VERIFY]"),
  p("fuel_supply_premium", "RQ4", "unif", 0, 0.5, "-", "[A] HALEU availability and regulatory premium on fission and NTR hardware"),
  p("fission_life_yr", "RQ4", "unif", 8, 12, "yr", "[S] FSP design life 10 years without maintenance"),
  p("venus_fission_shield_mult", "RQ4", "unif", 1.0, 1.5, "-", "[A] Tether / shadow-shield mass for a reactor hung below the city"),
  # dual-use water reserve
  p("water_reserve_kg", "RQ4", "unif", 750, 2500, "kg/colonist", "[D] 30-90 days x about 25 kg/day life-support reserve, doubles as wall mass"),
  p("export_kg_yr", "RQ4", "logunif", 0.01, 1, "kg/colonist/yr", "[A] Gram-scale exports"),
  p("export_value_usd_kg", "RQ4", "logunif", 1e4, 1e6, "USD/kg", "[VERIFY] Landed value of low-mass exports"),
  p("ntr_isp_s", "RQ4", "unif", 850, 950, "s", "[S] NTR on H2 (NERVA / DRACO class)"),
  p("ntr_dry_frac", "RQ4", "unif", 0.15, 0.25, "-", "[A] NTR + LH2 tankage dry fraction"),
  p("ntr_hw_usd_kg", "RQ4", "logunif", 5e3, 5e5, "USD/kg", "[A] Shuttle hardware cost incl. import"),
  p("ntr_reuse", "RQ4", "unif", 100, 300, "flights", "[A] Airframe reuse with local maintenance, both planets"),
  p("venus_ascent_dv_km_s", "RQ4", "unif", 8.0, 10.5, "km/s", "[D] Cloud level to LVO incl. losses"),
  p("mars_ascent_dv_km_s", "RQ4", "unif", 3.9, 4.5, "km/s", "[D] Surface to LMO incl. losses"),
  p("venus_h2_kWh_kg", "RQ4", "unif", 150, 450, "kWh/kg", "[D] Acid-derived H2O (about 9 kg/kg H2) + electrolysis + liquefaction"),
  p("mars_h2_kWh_kg", "RQ4", "unif", 70, 150, "kWh/kg", "[D] Ice-derived H2O + electrolysis + liquefaction"),
  p("tether_tip_km_s", "RQ4", "unif", 2.5, 3.5, "km/s", "[S] Rotating-tether tip speed (HASTOL 3.5 km/s), both planets"),
  p("venus_rockoon_save_km_s", "RQ4", "unif", 0.2, 0.6, "km/s", "[A] Balloon-lifted launch loss saving (Venus)"),
  p("venus_ramjet_v_km_s", "RQ4", "unif", 1.5, 3.0, "km/s", "[A] Nuclear ramjet on CO2 (Venus only; Mars air too thin)"),
  p("venus_wing_frac", "RQ4", "unif", 0.03, 0.10, "-", "[A] Wing/inlet dry fraction (Venus)"),
  p("venus_v_inf_km_s", "RQ4", "unif", 2.5, 5.0, "km/s", "[D] Venus-Earth v_inf, Hohmann to fast"),
  p("mars_v_inf_km_s", "RQ4", "unif", 2.65, 5.0, "km/s", "[D] Mars-Earth v_inf, Hohmann to fast"),
  p("venus_launch_capex_usd", "RQ4", "logunif", 2e5, 2e7, "USD/colonist", "[VERIFY] Tether / mass driver share"),
  p("mars_launch_capex_usd", "RQ4", "logunif", 5e4, 5e6, "USD/colonist", "[VERIFY] Ascent fleet share"),
  p("venus_transit_yr", "RQ4", "unif", 0.30, 0.42, "yr", "[S] Venus-Earth transit"),
  p("mars_transit_yr", "RQ4", "unif", 0.50, 0.75, "yr", "[S] Mars-Earth transit"),
  p("om_frac", "RQ4", "unif", 0.02, 0.08, "1/yr", "[A] O&M share of capex"),
  p("discount_rate", "RQ4", "unif", 0.03, 0.10, "1/yr", "[A] Real discount rate"),
  p("horizon_yr", "RQ4", "unif", 20, 40, "yr", "[A] NPV horizon"),
  # thermal control
  p("mars_ambient_C", "RQ4", "unif", -70, -50, "C", "[S] Mean surface temperature about -63 C"),
  p("k_mars_kW_K", "RQ4", "unif", 0.05, 0.40, "kW/K/colonist", "[A] Conductance of habitat + greenhouse envelope (about 50-200 m2 grow area per colonist)"),
  p("k_venus_kW_K", "RQ4", "unif", 0.05, 0.30, "kW/K/colonist", "[A] Conductance, dense 1 bar convective coupling"),
  p("internal_heat_frac", "RQ4", "unif", 0.2, 0.6, "-", "[A] Share of electric load dissipated inside the pressurized volume"),
  p("draw_duty", "RQ4", "unif", 0.05, 0.30, "-", "[A] Share of daylight hours the whole city spends at draw depth", "city"),
  p("draw_alt_km", "RQ4", "unif", 47.5, 50.5, "km", "[S] Draw depth in the lower cloud", "city"),
  p("carnot_frac", "RQ4", "unif", 0.35, 0.6, "-", "[S] Practical heat pump COP as fraction of Carnot"),
  # fertilizer make-up
  p("biomass_dry_kg_yr", "RQ4", "unif", 500, 900, "kg/colonist/yr", "[A] Edible + inedible dry crop biomass at full substitution"),
  p("nutrient_recycle", "RQ4", "unif", 0.85, 0.98, "-", "[A] Black water + inedible biomass recycled to hydroponics (both planets) [VERIFY vs MELiSSA / BLSS]"),
  p("venus_surface_mineral_usd_kg", "RQ4", "logunif", 500, 5e5, "USD/kg", "[A] Robotic surface excavation (460 C, 92 bar; SiC electronics demonstrated about 30 days in GEER) + balloon ore return + station processing [VERIFY]"),
  p("venus_N_kWh_kg", "RQ4", "unif", 10, 15, "kWh/kg N", "[A] Haber-Bosch with electrolytic H2, N2 is 3.5% of Venus air"),
  p("mars_N_kWh_kg", "RQ4", "unif", 25, 80, "kWh/kg N", "[A] N2 from 6 mbar air at about 2.8% [VERIFY]"),
  p("mars_PK_kWh_kg", "RQ4", "unif", 5, 40, "kWh/kg", "[A] P and K leaching from regolith incl. perchlorate cleanup"),
  p("mars_S_other_kWh_kg", "RQ4", "unif", 2, 10, "kWh/kg", "[A] Sulfate and Ca/Mg/Fe/micros from regolith"),
  gravity_block("RQ4"),
  p("labor_value_usd_hr", "RQ4", "logunif", 1e3, 3e4, "USD/h", "[A] Value of a productive colonist hour off-world"),
  p("replace_usd", "RQ4", "logunif", 1e6, 5e7, "USD", "[A] Cost to replace a lost colonist (transport, training)"),
  # hydroponic lighting and CO2
  p("crop_area_m2", "RQ4", "unif", 40, 100, "m2/colonist", "[A] Growing area at full food substitution [VERIFY vs NASA BLSS]"),
  p("target_DLI", "RQ4", "unif", 20, 40, "mol/m2/day", "[S] Typical crop daily light integral targets"),
  p("led_umol_J", "RQ4", "unif", 2.5, 3.5, "umol/J", "[S] Current horticultural LED efficacy"),
  p("led_kg_per_kW", "RQ4", "unif", 10, 30, "kg/kW", "[A] LED fixture + driver mass"),
  p("glazing_trans", "RQ4", "unif", 0.6, 0.85, "-", "[A] Greenhouse envelope PAR transmission"),
  p("mars_waterwall_par", "RQ4", "unif", 0.6, 0.85, "-", "[D] PAR transmission through 0.4-1 m water wall (red absorbed most)"),
  p("venus_night_light_frac", "RQ4", "unif", 0, 0.3, "-", "[A] Share of the 50 h night lit at 1/4 target for photoperiod-sensitive crops"),
  p("co2_kg_per_kg_dry", "RQ4", "unif", 1.4, 1.8, "kg/kg", "[D] CH2O stoichiometry 44/30 = 1.47 plus losses"),
  p("mars_co2_kWh_kg", "RQ4", "unif", 0.1, 0.3, "kWh/kg", "[D] Compress 6 mbar to 1 bar: 0.06 kWh/kg isothermal ideal, x2-5 real"),
  # gas-fermentation protein (both planets)
  p("protein_kg_yr", "RQ4", "unif", 25, 40, "kg/colonist/yr", "[S/D] About 0.8-1.2 g/kg/day for working adults"),
  p("ferm_share", "RQ4", "unif", 0.3, 0.8, "-", "[A] Share of protein from gas fermentation"),
  p("protein_kWh_kg", "RQ4", "unif", 18, 85, "kWh/kg protein", "[S] Thermodynamic floor to Smaje 2024 estimate"),
  p("protein_frac", "RQ4", "unif", 0.60, 0.70, "-", "[S] Protein share of microbial biomass"),
  p("ferm_kg_per_kgyr", "RQ4", "logunif", 0.05, 0.5, "kg per kg/yr", "[A] Bioreactor plant mass per kg/yr [VERIFY]"),
  # water-wall shielding
  p("mars_wall_area_m2", "RQ4", "unif", 50, 200, "m2/colonist", "[A] Whole facility incl. greenhouse envelope (water walls stay transparent)"),
  p("mars_wall_g_cm2", "RQ4", "unif", 40, 100, "g/cm2", "[A] Areal density to cut GCR + SEP meaningfully [VERIFY]"),
  p("venus_shelter_area_m2", "RQ4", "unif", 10, 30, "m2/colonist", "[A] Crew-quarters storm shelter only"),
  p("venus_shelter_g_cm2", "RQ4", "unif", 5, 40, "g/cm2", "[D/A] SEP shelter; RQ1 physics puts Venus cloud-level SEP dose near zero"),
  p("venus_lift_penalty", "RQ4", "unif", 1.05, 1.2, "-", "[A] Extra envelope and structure to float the water mass"),
  # isotope co-products of acid / ice mining (v17)
  p("isru_water_kg_yr", "RQ4", "logunif", 500, 10000, "kg/colonist/yr", "[A] Water processed per colonist-year (life-support make-up, O2 feed, NTR propellant), both planets"),
  p("venus_air_kg_per_kg_water", "RQ4", "logunif", 1.6e5, 7.7e5, "kg air/kg water", "[D] RQ2 collector physics, 10th-90th percentile"),
  p("venus_He_ppm", "RQ4", "logunif", 4, 36, "ppm (molar)", "[S] Pioneer Venus about 12 (+24/-8) ppm He"),
  p("he3_ratio", "RQ4", "logunif", 1.4e-6, 3e-4, "3He/4He", "[S/A] Earth air to the Pioneer Venus upper limit 3e-4 [VERIFY by probe]"),
  p("he_recovery", "RQ4", "unif", 0.3, 0.9, "-", "[A] Share of collector-air He captured"),
  p("he_kWh_kg_air", "RQ4", "logunif", 2e-4, 5e-2, "kWh/kg air", "[A] He separation from collector exhaust [VERIFY]"),
  p("he4_kWh_kg", "RQ4", "unif", 5, 20, "kWh/kg He", "[A] Cryogenic 3He/4He separation per kg bulk He"),
  p("he_capex_mult", "RQ4", "unif", 0.2, 1.0, "-", "[A] He plant capital and upkeep / energy cost"),
  p("he3_price_usd_kg", "RQ4", "logunif", 5e6, 1.11e8, "USD/kg", "[S/A] $15M-111M/kg market; floor $5M for supply response"),
  p("venus_DH", "RQ4", "unif", 0.0156, 0.025, "D/H", "[S] Venus water 100-150 x VSMOW"),
  p("mars_DH", "RQ4", "unif", 7.8e-4, 1.25e-3, "D/H", "[S] Mars water about 5-8 x VSMOW (atmosphere about 7; polar ice modeled about 8)"),
  p("d2o_recovery", "RQ4", "unif", 0.5, 0.9, "-", "[A] Feed deuterium recovered as D2O, both planets"),
  p("d2o_ref_kWh_kg", "RQ4", "logunif", 2000, 33000, "kWh/kg D2O", "[D] Natural-feed reference energy, scaled by feed D content"),
  p("d2o_synergy", "RQ4", "unif", 0, 0.7, "-", "[A] Enrichment carried by existing NTR-H2 electrolysis, both planets"),
  p("d2o_capex_mult", "RQ4", "unif", 0.3, 1.0, "-", "[A] Enrichment plant capital and upkeep / energy cost"),
  p("d2o_price_usd_kg", "RQ4", "logunif", 270, 1000, "USD/kg", "[S] About $270-300/kg natural-feed heavy water; upper end [A]"),
  p("earth_launch_usd_kg", "RQ4", "logunif", 100, 7000, "USD/kg", "[S/A] Earth launch to LEO (D2O cislunar parity price)"),
  # settlement motive (v18)
  p("colonist_comp_usd_yr", "RQ4", "logunif", 1e5, 5e5, "USD/colonist/yr", "[A] Colonist compensation before hardship premium, both planets"),
  p("venus_hardship_frac", "RQ4", "unif", 0.10, 0.45, "-", "[S/A] US post hardship and danger pay each 5-35% of base pay; Venus: 1 atm outdoors with respirator and acid gear, lethal surface below, short trip"),
  p("mars_hardship_frac", "RQ4", "unif", 0.20, 0.70, "-", "[S/A] Same scales; Mars: pressure suit for every step outside, longer trip, weaker sunlight (gravity excluded, in RQ5)"),
  p("supply_kg_yr", "RQ4", "logunif", 100, 10000, "kg/colonist/yr", "[A] Bulk volatile supply to cislunar buyers, both planets"),
  p("venus_N2_kWh_kg", "RQ4", "unif", 2, 15, "kWh/kg N2", "[A] N2 (3.5% of Venus air) by CO2 condensation; shares the He-recovery gas stream [VERIFY]"),
  p("supply_capex_mult", "RQ4", "unif", 0.2, 1.0, "-", "[A] Venus N2 plant capital and upkeep / energy cost (Mars water price already all-in)"),
  # He-3 market depth and national-security cost share (v19)
  p("colony_size_n", "RQ4", "logunif", 100, 10000, "colonists", "[A] Colony population; scales total He-3 supply against the market"),
  p("he3_market_kg_yr", "RQ4", "logunif", 1, 20, "kg/yr", "[S/A] Present demand about 2 kg/yr (federal < 6,000 L/yr, DOE Isotope Program; Bluefors up to 10,000 L/yr, 2028-2037); upper end allows quantum-computing growth"),
  p("he3_elasticity", "RQ4", "unif", 0.5, 1.5, "-", "[A] Price elasticity of He-3 demand"),
  p("moon_share", "RQ4", "unif", 0, 0.8, "-", "[A] Market share held by lunar (Interlune) and terrestrial tritium-decay He-3"),
  p("gov_cost_share", "RQ4", "unif", 0, 0.3, "-", "[A] National-security share of startup capital, paid only to a colony supplying He-3 or D2O at a profit (DOE bought lunar He-3, 2025; no US heavy water production since 1982)"),
  # research-contract income (v20)
  p("research_usd_yr", "RQ4", "logunif", 5e7, 1e9, "USD/colony/yr", "[S/A] Government research program per colony; anchor: NSF Antarctic Facilities and Operations about $244M (FY2023 request) for about 150 projects per season"),
  p("venus_science_mult", "RQ4", "unif", 0.5, 1.5, "-", "[A] Venus science value: 1g control lab, surface engineering, propulsion test bed, greenhouse and cloud science"),
  p("mars_science_mult", "RQ4", "unif", 0.5, 1.5, "-", "[A] Mars science value: subsurface life search, ice record, accessible geology (same range as Venus)"),
  # cloud-level weather (v24)
  p("conv_top_km", "RQ4", "unif", 55.5, 59, "km", "[S] Top of the main convective layer: model about 55.5 (Lefevre et al. 2018), VeRa 49-59, Akatsuki 50-58"),
  p("venus_gust_capex_frac", "RQ4", "unif", 0.03, 0.12, "-", "[A] Gust-load structure and active altitude control as a share of capital when floating inside the convective layer (VEGA downdrafts to about 3.5 m/s); 30% of this above it"),
  p("envelope_capex_frac", "RQ4", "unif", 0.05, 0.20, "-", "[A] Envelope and lift-gas share of capital at the 52.5 km reference density"),
  p("mars_water_usd_kg", "RQ4", "logunif", 32.5, 2830, "USD/kg", "[D] RQ2 v8 physics, water-only, 10th-90th percentile"),
  p("venus_water_usd_kg", "RQ4", "logunif", 171, 9600, "USD/kg", "[D] RQ2 v8 physics (section draw), water-only, 10th-90th percentile")
)
# Note: colony_load_kW enters RQ2 only in hold mode (propulsion overhead); RQ4
# uses it in both modes (power block row for hold, explicit row for drift).

PARAMS <- rbind(PARAMS, gravity_block("RQ5"), depart_block("RQ3"), depart_block("RQ4"),
                power_block("RQ3"))
PARAMS <- PARAMS[!(PARAMS$rq == "RQ3" & PARAMS$mode == "drift"), ]

PRODUCTS <- data.frame(
  product   = c("paclitaxel", "vinca_alkaloid", "cannabis_flower", "thc_distillate", "cbg_isolate", "thcv_isolate"),
  class     = c("pharma", "pharma", "cannabis", "cannabis", "cannabis", "cannabis"),
  lo        = c(15000, 100000, 660, 1000, 1000, 4000),
  hi        = c(30000, 300000, 3300, 5000, 1500, 8500),
  bio_lo    = c(90, 20000, 1.1, 4, 5, 20),
  bio_hi    = c(900, 300000, 1.5, 8, 15, 200),
  extract   = c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE),
  source    = c("[S] Pharmaoffer avg $20.7k/kg API (12 mo to Jun 2026); biomass via plant cell culture [VERIFY yield]",
                "[S] PharmaCompass about $197k/kg vincristine sulfate (undated customs avg); leaf yield about 0.0003-0.005% [VERIFY]",
                "[S] US wholesale 'several hundred to over a thousand dollars' per lb (2026)",
                "[VERIFY] Distillate wholesale; about 15-25% THC flower",
                "[S] KV Cannabis CBG isolate from $1,250/kg (2026); CBG-dominant strains about 7-20%",
                "[S] Red Mesa Science THCV isolate $832/100 g to about $4,875/kg; THCV content about 0.5-5%"),
  stringsAsFactors = FALSE)
CROPS <- PRODUCTS   # legacy name used by the sampler

# Plant tissue nutrient mass fractions of dry biomass [S: typical hydroponic crop tissue, VERIFY]
NUTRIENT <- c(N = 0.040, P = 0.005, K = 0.035, S = 0.003, other = 0.015)
VENUS_TOA_WM2 <- 2614                 # [S] Landis & Colozza Table 1
SYNODIC_VENUS_D <- 583.9; SYNODIC_MARS_D <- 779.9
H2SO4_H2O <- 18.015 / 98.08           # kg H2O per kg H2SO4
H2SO4_O2  <- 15.999 / 98.08           # kg O2 per kg H2SO4 (1/2 O2 per SO3)
VSMOW_DH  <- 1.5576e-4                # Earth ocean D/H
D2O_PER_KG_WATER <- function(DH) 1.1117 * DH / (1 + DH)   # kg D2O per kg water at full recovery
HE3_MASS_FRAC <- function(ppm, ratio) ppm * 1e-6 * ratio * 3.016 / 43.45   # kg He-3 per kg Venus air
HE_MASS_FRAC  <- function(ppm) ppm * 1e-6 * 4.003 / 43.45                  # kg He per kg Venus air

# -----------------------------------------------------------------------------
# Sampling and feature space
# -----------------------------------------------------------------------------
draw_params <- function(tab, n) {
  out <- list(); crop <- NULL
  for (i in seq_len(nrow(tab))) {
    a <- tab$a[i]; b <- tab$b[i]
    out[[tab$name[i]]] <- switch(tab$dist[i],
      unif    = runif(n, a, b),
      logunif = 10^runif(n, log10(a), log10(b)),
      crop    = {
        k <- sample(seq_len(nrow(CROPS)), n, replace = TRUE)
        crop <- CROPS$crop[k]
        10^runif(n, log10(CROPS$lo[k]), log10(CROPS$hi[k]))
      },
      product = {
        k <- sample(seq_len(nrow(PRODUCTS)), n, replace = TRUE)
        crop <- PRODUCTS$product[k]
        10^runif(n, log10(PRODUCTS$lo[k]), log10(PRODUCTS$hi[k]))
      })
  }
  d <- as.data.frame(out)
  if (!is.null(crop)) d$crop <- crop
  d
}

to_features <- function(d, tab) {
  X <- d[, tab$name, drop = FALSE]
  for (i in seq_len(nrow(tab)))
    if (tab$dist[i] %in% c("logunif", "crop", "product")) X[[tab$name[i]]] <- log10(X[[tab$name[i]]])
  # categorical inputs as true factors (Eigen-encoded in H2O)
  if ("product_price_usd_kg" %in% tab$name)
    X$product <- factor(d$crop, levels = PRODUCTS$product)
  if ("market_u" %in% tab$name) {
    X$market <- factor(ifelse(d$market_u >= 0.5, "mars_orbit", "leo"), levels = c("leo", "mars_orbit"))
    X$market_u <- NULL
  }
  X
}
FACTOR_INFO <- c(product = "RQ3 product identity (factor, Eigen-encoded)",
                 market  = "RQ3 protein market: LEO/cislunar vs Mars orbit (factor, Eigen-encoded)")

# -----------------------------------------------------------------------------
# Physics helpers
# -----------------------------------------------------------------------------
# VIRA reference atmosphere (Seiff et al. 1985), 45-65 km
VIRA <- data.frame(z = c(45, 50, 55, 60, 65), T = c(385.4, 350.5, 302.3, 262.8, 243.2),
                   P = c(1.979, 1.066, 0.5314, 0.2357, 0.09765))           # K, bar
venus_T_K <- function(alt) approx(VIRA$z, VIRA$T, alt, rule = 2)$y
venus_P_bar <- function(alt) exp(approx(VIRA$z, log(VIRA$P), alt, rule = 2)$y)
venus_rho <- function(alt) venus_P_bar(alt) * 1e5 * 0.04401 / (8.314 * venus_T_K(alt))
venus_rho_old <- function(alt) {
  f <- (alt - 50) / 5
  P <- 1.066 * (0.531 / 1.066)^f * 101325
  P * 0.04401 / (8.314 * venus_T_K(alt))
}
cloud_loading_mg_m3 <- function(alt, spread)          # K&H 1980: about 21 mg/m3 lower cloud
  21 * (7.5 / 21)^((alt - 49) / 4.5) * 10^spread          #  centre (49 km), 7.5 middle cloud (53.5 km)
droplet_um <- function(alt, f)                            # mass-weighted: mode 3 dominates below 50.5 km
  (2.5 + 4.5 * plogis((50.5 - alt) / 0.5)) * f
temp_factor <- function(Tc, coef) pmax(1 + coef * (Tc - 25), 0.1)
deg <- pi / 180

# Ice mass fraction of mined material: pore ice / hydrated regolith (about
# 2-30%) at lower latitudes, grading to near-pure excess ice (about 50-95%)
# above about 50 deg. u in [0, 1] positions the draw log-uniformly in the range.
mars_ice_frac <- function(lat, u) {
  s <- plogis((lat - 50) / 2)
  lo <- 0.02 + (0.50 - 0.02) * s
  hi <- 0.30 + (0.95 - 0.30) * s
  10^(log10(lo) + u * (log10(hi) - log10(lo)))
}

# Returns kg of power system per W of useful average load, for ISRU (no
# storage, daylight-only plants) and for the colony (storage for life support).
power_model <- function(d, mode) {
  mars_geom <- (1 / pi) * cos(d$mars_site_lat * deg)         # sol-average of cos(zenith)
  mars_Wm2 <- d$mars_toa_Wm2 * d$mars_tau_trans * d$mars_storm_factor * d$mars_panel_dust *
              mars_geom * d$cell_eff * temp_factor(d$mars_cell_C, d$cell_temp_coef) * d$pc_eff
  mars_arr <- d$array_kg_m2 / mars_Wm2
  store_kg_per_W <- function(night_h)                         # lighter of battery or RFC, per W of night load
    pmin(night_h / (d$battery_Wh_kg * d$dod), night_h / d$rfc_Wh_kg + d$rfc_stack_kg_per_kW / 1000)
  mars_sto <- d$ls_frac * store_kg_per_W(d$mars_night_h)
  # solar-only colony must carry its critical (night/base) load through a
  # weeks-long global dust storm: oversize the array to cover it
  mars_arr_storm <- pmax(mars_arr, mars_arr * d$ls_frac / d$mars_storm_output)

  Tcell_v <- venus_T_K(d$venus_alt_km) - 273.15 + 10
  # Drifting with the superrotation: half the cycle lit, cos-averaged
  geom <- 0.5 * (2 / pi) * cos(d$drift_lat * deg)
  night_h <- d$drift_night_h * cos(d$drift_lat * deg); dayfrac <- 0.5; Pprop_W <- 0
  light_loss <- ifelse(d$venus_alt_km < 56.5, d$disrupt_light_loss, 0)
  venus_Wm2 <- VENUS_TOA_WM2 * d$venus_cloud_trans * (1 - light_loss) * geom * d$bifacial_gain * d$cell_eff *
               temp_factor(Tcell_v, d$cell_temp_coef) * d$pc_eff
  over <- 1
  venus_arr <- d$array_kg_m2 / venus_Wm2 * over
  venus_sto <- d$ls_frac * store_kg_per_W(night_h)
  list(mars_isru = mars_arr, mars_colony = mars_arr_storm + mars_sto, mars_dayfrac = 0.5,
       venus_isru = venus_arr, venus_colony = venus_arr + venus_sto, venus_dayfrac = dayfrac,
       mars_Wm2 = mars_Wm2, venus_Wm2 = venus_Wm2, Pprop_kW = Pprop_W / 1000,
       mars_arr = mars_arr, venus_arr = venus_arr, venus_night_h = night_h)
}

usd_per_kWh <- function(kg_per_W, tc, life_yr) kg_per_W * tc * 1000 / (8766 * life_yr)

# -----------------------------------------------------------------------------
# Models: each returns D (positive favors H1) plus the two arms
# -----------------------------------------------------------------------------
rq1_model <- function(d, mode) {
  k <- qpois(pmin(d$sep_count_u, 1 - 1e-9), d$sep_rate_yr)      # large SEP events this year
  sev <- k * 10^d$sep_scatter                                   # deterministic in the inputs
  shelter <- d$shelter_warning * d$shelter_transmission + (1 - d$shelter_warning)
  mars_sep <- d$mars_sep_event_mGy * sev * d$sep_Q * shelter
  venus_sep <- d$venus_sep_event_mGy * d$venus_dose_growth_per_km^(d$venus_alt_km - 51) *
               sev * d$sep_Q * shelter
  mars_gcr <- d$mars_gcr_mGy_day * 365.25 * d$mars_Q
  mix <- d$mars_time_inside * d$mars_hab_transmission + (1 - d$mars_time_inside)
  mars <- mars_gcr * mix + mars_sep
  venus <- d$venus_gcr51_mGy_h * d$venus_dose_growth_per_km^(d$venus_alt_km - 51) *
           8766 * d$venus_Q + venus_sep
  list(D = mars - venus, A = venus, B = mars,
       D_lab = "Mars minus Venus dose equivalent (mSv/yr)",
       extra = data.frame(sep_events = k, mars_sep_mSv = mars_sep, venus_sep_mSv = venus_sep,
                          mars_gcr_mSv = mars_gcr * mix))
}

rq2_model <- function(d, mode) {
  pw <- power_model(d, mode)
  w <- d$basket_h2o_frac
  ice <- mars_ice_frac(d$mars_site_lat, d$mars_ice_u)
  material <- (1 / ice) * (1 + d$mars_strip_ratio)                       # kg moved per kg water
  E_dig <- material * d$mars_dig_kWh_kg * (1 + d$mars_cold_penalty)
  E_heat <- ((1 / ice - 1) * 800 * d$mars_dT_K + 2100 * d$mars_dT_K + 2.83e6) / 3.6e6 *
            (1 - d$mars_heat_recovery) / d$mars_extract_eff
  mars_E_h2o <- E_dig + E_heat + d$mars_purify_kWh_kg
  mars_E <- w * mars_E_h2o + (1 - w) * d$mars_o2_kWh_kg
  fleet_kg_per_kgyr <- w * material / d$mars_excavator_tput                # per kg/yr basket

  # Venus acid-primary route, per kg of basket
  alt_c   <- d$collector_alt_km
  rho_c   <- venus_rho(alt_c)
  load_c  <- cloud_loading_mg_m3(alt_c, d$aerosol_spread)
  dia     <- droplet_um(alt_c, d$droplet_size_factor)
  cap_eff <- 1 / (1 + (d$pad_d50_um / dia)^d$pad_sharpness)
  dP      <- d$pad_K * 0.5 * rho_c * d$face_v_m_s^2
  sol_m3  <- cap_eff * load_c * 1e-6                                       # kg solution per m3
  fan_kWh_kg_sol <- (dP / d$fan_eff / 3.6e6) / sol_m3
  y_w <- (1 - d$venus_acid_wt) + d$venus_acid_wt * H2SO4_H2O              # kg H2O per kg solution
  y_o <- d$venus_acid_wt * H2SO4_O2                                       # kg O2 per kg solution
  S <- w / y_w
  o2_acid <- pmin(S * y_o, 1 - w)
  o2_soxe <- (1 - w) - o2_acid
  venus_E <- S * fan_kWh_kg_sol +
             S * (1 - d$venus_acid_wt) * d$evap_kWh_kg +
             S * d$venus_acid_wt * d$dehyd_kWh_kg_acid +
             o2_acid * d$so3_decomp_kWh_kg_O2 +
             o2_soxe * d$venus_o2_kWh_kg

  # ---- process equipment mass per kg/yr of basket (daylight-only operation) ----
  h_op <- 8766 * 0.5                                   # processing at habitat level, daylight
  h_col <- h_op * d$draw_duty * (1 - d$disrupt_collect_loss)  # collection only at draw depth, less disruption passages
  air_m3_per_kg <- S / sol_m3
  flow_m3_s <- air_m3_per_kg / (h_col * 3600)
  face_m2 <- flow_m3_s / d$face_v_m_s
  hull <- if (mode == "section") d$section_hull_mult * d$tether_hull_mult else 1
  wind_driven <- if (mode == "section") d$shear_wind_m_s >= d$face_v_m_s else rep(FALSE, nrow(d))
  buffer_kg <- S * (1 - d$draw_duty) / d$draws_per_yr  # acid stored between draws, per kg/yr basket
  venus_equip <- d$bop_mult * (
      hull * (face_m2 * d$collector_kg_m2 +
              ifelse(wind_driven, 0, flow_m3_s * dP / d$fan_eff / d$fan_W_per_kg)) +
      buffer_kg * d$tank_kg_per_kg +
      S / h_op * d$reactor_kg_per_kgh +
      w / h_op * d$condenser_kg_per_kgh +
      o2_soxe / h_op * d$soxe_kg_per_kgh)
  mars_equip <- d$bop_mult * (
      w * E_heat / h_op * d$oven_kg_per_kW +
      w / h_op * d$condenser_kg_per_kgh +
      (1 - w) / h_op * d$soxe_kg_per_kgh * d$mars_compressor_factor)

  life_h <- 8766 * d$life_yr
  cost <- function(E, kgW, equip, spares, mult)
    (E * 1000 * kgW / life_h + equip * (1 / d$life_yr + spares)) * d$transport_base_usd_kg * mult
  mars  <- cost(mars_E,  pw$mars_isru,  mars_equip, d$mars_process_spares, d$mars_transport_mult) +
           fleet_kg_per_kgyr / pw$mars_dayfrac * (1 / d$life_yr + d$mars_spares_frac) *
           d$transport_base_usd_kg * d$mars_transport_mult
  fan_E <- S * fan_kWh_kg_sol
  venus_E_eff <- venus_E + fan_E * (1 / d$depth_irr_factor - 1)   # arrays under the cloud deck
  venus_E_eff <- ifelse(wind_driven, venus_E_eff - fan_E * (1 / d$depth_irr_factor), venus_E_eff)  # wind does the work
  venus <- cost(venus_E_eff, pw$venus_isru, venus_equip, d$venus_spares_frac, d$venus_transport_mult)
  list(D = mars - venus, A = venus, B = mars,
       D_lab = "Mars minus Venus delivered cost (USD/kg basket)",
       extra = data.frame(mars_ice_frac = ice, mars_kg_moved_per_kg_water = material,
                          mars_E_h2o_kWh_kg = mars_E_h2o, mars_fleet_kg_per_kgyr = fleet_kg_per_kgyr,
                          mars_E_kWh_kg = mars_E, venus_E_kWh_kg = venus_E,
                          collector_T_C = venus_T_K(alt_c) - 273.15, collector_P_atm = venus_P_bar(alt_c) / 1.01325,
                          cloud_mg_m3 = load_c, droplet_um = dia, capture_eff = cap_eff, pad_dP_Pa = dP,
                          air_m3_per_kg_basket = air_m3_per_kg,
                          venus_equip_kg_per_kgyr = venus_equip, mars_equip_kg_per_kgyr = mars_equip,
                          venus_E_with_depth_penalty = venus_E_eff, wind_driven = wind_driven,
                          venus_fan_share = S * fan_kWh_kg_sol / venus_E,
                          venus_o2_from_acid_share = o2_acid / (1 - w),
                          so2_byproduct_kg = o2_acid * 64.066 / 15.999,
                          mars_Wm2_avg = pw$mars_Wm2, venus_Wm2_avg = pw$venus_Wm2,
                          mars_isru_kg_per_W = pw$mars_isru, venus_isru_kg_per_W = pw$venus_isru,
                          venus_prop_kW_per_colonist = pw$Pprop_kW))
}

stage_pf <- function(dv_m_s, isp, f) pmax(exp(-dv_m_s / (isp * 9.80665)) - f, 0)

rq3_protein <- function(d) {
  pw <- power_model(d, "section")
  usd_kWh <- pw$venus_isru * d$transport_base_usd_kg * d$venus_transport_mult * 1000 / (8766 * d$life_yr)
  # Venus production per kg product
  prodV <- d$protein_frac * d$protein_kWh_kg * usd_kWh +
           d$protein_frac * 0.16 * d$N_kWh_kg * usd_kWh +
           d$ferm_kg_per_kgyr * d$transport_base_usd_kg * d$venus_transport_mult / d$life_yr
  # Venus cloud level to LVO (full chain), then mass-driver departure, aerocapture at market
  is_mars <- d$market_u >= 0.5
  dv <- pmax(d$ascent_dv_km_s - d$rockoon_save_km_s - d$ramjet_v_km_s - d$tether_tip_km_s, 0.5)
  f <- d$dry_frac + d$wing_frac
  pf <- stage_pf(dv * 1000, d$isp_s, f)
  cap <- capsule_frac(d)
  m0 <- 1 / (pf * cap); dry <- f * m0 + (1 / cap - 1); prop <- m0 - dry - 1
  hw <- dry * (d$imported_frac_dry * d$hardware_usd_kg / d$vehicle_reuse +
               d$reactor_frac_dry * d$hardware_usd_kg * (1 + d$fuel_supply_premium) / pmin(d$reactor_life_flights, d$vehicle_reuse) +
               (1 - d$imported_frac_dry - d$reactor_frac_dry) * d$local_mfg_usd_kg / d$vehicle_reuse)
  vinfV <- ifelse(is_mars, d$venus_v_inf_mars, d$venus_v_inf_km_s)
  departV <- departure_cost(d, MU_VENUS, R_LVO, vinfV, SUN_VENUS) / cap
  venus_cost <- prodV + prop * d$prop_kWh_kg * usd_kWh + hw + departV + d$tether_fee_usd_kg
  feasible <- is.finite(m0) & pf > 0
  venus_cost[!feasible] <- CFG$infeasible_cost_usd_kg
  # Earth supply: launch to LEO; for Mars, mass-driver departure from LEO and aerocapture
  departE <- departure_cost(d, 3.986e14, (6378 + 400) * 1000, d$earth_v_inf_mars, 1)
  earth_cost <- d$earth_protein_usd_kg +
                ifelse(is_mars, (d$earth_launch_usd_kg + departE) / cap, d$earth_launch_usd_kg)
  list(D = earth_cost - venus_cost, A = venus_cost, B = earth_cost,
       D_lab = "Earth-supplied minus Venus-delivered protein cost (USD/kg)",
       extra = data.frame(product = ifelse(is_mars, "protein_to_mars_orbit", "protein_to_LEO"),
                          venus_production_usd_kg = prodV, venus_delivered_usd_kg = venus_cost,
                          earth_delivered_usd_kg = earth_cost, station_usd_kWh = usd_kWh,
                          liftoff_kg_per_kg = ifelse(feasible, m0, NA), feasible = feasible))
}

# Throughput-scaled transport capital: sets the per-kg tether fee and station charge
scale_transport <- function(d, transport_mult) {
  T_t <- d$colony_size_n * d$traffic_kg_colonist_yr / 1000                  # t/yr through the system
  ref_tether <- 14 * d$catches_per_yr                                       # t/yr, HASTOL-class reference
  fee_ref <- d$tether_mass_ratio * d$tether_unit_mult * d$transport_base_usd_kg * transport_mult *
             d$capital_charge / d$catches_per_yr                            # USD/kg at reference capacity
  list(tether = fee_ref * (T_t / ref_tether)^(d$scale_exp - 1),
       station = d$station_ref_usd_kg * (T_t / 1000)^(d$scale_exp - 1),
       traffic_t_yr = T_t)
}

# Capsule product share from container slenderness (shield ~ frontal area, product ~ volume)
capsule_frac <- function(d, v_entry = NULL) {
  f <- d$container_LD / (d$container_LD + d$aero_shell_c)
  if (!is.null(v_entry)) f <- f - d$entry_speed_penalty * pmax(v_entry - 11.5, 0)
  pmax(f, 0.2)
}

# Venus full launch chain (balloon, ramjet, tether, single-stage NTR-H2) to LVO,
# then mass-driver departure; returns USD per kg delivered (cap = payload share
# of the entry / aerocapture capsule)
venus_chain_cost <- function(d, usd_kWh, v_inf, cap) {
  dv <- pmax(d$ascent_dv_km_s - d$rockoon_save_km_s - d$ramjet_v_km_s - d$tether_tip_km_s, 0.5)
  f <- d$dry_frac + d$wing_frac
  pf <- stage_pf(dv * 1000, d$isp_s, f)
  m0 <- 1 / (pf * cap); dry <- f * m0 + (1 / cap - 1); prop <- m0 - dry - 1
  hw <- dry * (d$imported_frac_dry * d$hardware_usd_kg / d$vehicle_reuse +
               d$reactor_frac_dry * d$hardware_usd_kg * (1 + d$fuel_supply_premium) / pmin(d$reactor_life_flights, d$vehicle_reuse) +
               (1 - d$imported_frac_dry - d$reactor_frac_dry) * d$local_mfg_usd_kg / d$vehicle_reuse)
  out <- prop * d$prop_kWh_kg * usd_kWh + hw + departure_cost(d, MU_VENUS, R_LVO, v_inf, SUN_VENUS) / cap +
         d$tether_fee_usd_kg
  out[!(is.finite(m0) & pf > 0)] <- CFG$infeasible_cost_usd_kg
  out
}

# RQ3 isotope scenario: net co-product margin per tonne of acid-derived water
rq3_isotope <- function(d) {
  pw <- power_model(d, "section")
  usd_kWh <- pw$venus_isru * d$transport_base_usd_kg * d$venus_transport_mult * 1000 / (8766 * d$life_yr)
  # collector air pulled per kg of acid-derived water (RQ2 physics)
  alt_c <- d$collector_alt_km
  dia <- droplet_um(alt_c, d$droplet_size_factor)
  cap_eff <- 1 / (1 + (d$pad_d50_um / dia)^d$pad_sharpness)
  sol_m3 <- cap_eff * cloud_loading_mg_m3(alt_c, d$aerosol_spread) * 1e-6
  y_w <- (1 - d$venus_acid_wt) + d$venus_acid_wt * H2SO4_H2O
  air_kg <- venus_rho(alt_c) / (sol_m3 * y_w)                       # kg air per kg water
  # helium-3 from the collector exhaust, shipped to Earth
  he3_kg <- air_kg * HE3_MASS_FRAC(d$venus_He_ppm, d$he3_ratio) * d$he_recovery
  he_kg  <- air_kg * HE_MASS_FRAC(d$venus_He_ppm) * d$he_recovery
  he_cost <- (air_kg * d$he_kWh_kg_air + he_kg * d$he4_kWh_kg) * usd_kWh * (1 + d$he_capex_mult)
  ship_he <- venus_chain_cost(d, usd_kWh, d$venus_v_inf_km_s,
                              capsule_frac(d, sqrt(d$venus_v_inf_km_s^2 + 11.18^2))) + d$downmass_usd_kg
  he_margin <- he3_kg * (d$he3_price_usd_kg - ship_he) - he_cost
  # heavy water from the acid-derived water, sold in cislunar space
  d2o_kg <- D2O_PER_KG_WATER(d$venus_DH) * d$d2o_recovery
  e_d2o <- d$d2o_ref_kWh_kg * (VSMOW_DH / d$venus_DH) * (1 - d$d2o_synergy)   # kWh per kg D2O
  d2o_cost <- d2o_kg * e_d2o * usd_kWh * (1 + d$d2o_capex_mult)
  ship_d2o <- venus_chain_cost(d, usd_kWh, d$venus_v_inf_km_s, capsule_frac(d))
  d2o_margin <- d2o_kg * (d$d2o_price_usd_kg + d$earth_launch_usd_kg - ship_d2o) - d2o_cost
  D <- 1000 * (he_margin + d2o_margin)
  list(D = D, A = 1000 * (he3_kg * d$he3_price_usd_kg + d2o_kg * (d$d2o_price_usd_kg + d$earth_launch_usd_kg)),
       B = 1000 * (he3_kg * ship_he + he_cost + d2o_kg * ship_d2o + d2o_cost),
       D_lab = "Net He-3 + D2O co-product margin (USD per tonne acid-derived water)",
       extra = data.frame(air_kg_per_kg_water = air_kg, he3_g_per_t_water = 1e6 * he3_kg,
                          d2o_kg_per_t_water = 1000 * d2o_kg, he3_margin_usd_t = 1000 * he_margin,
                          d2o_margin_usd_t = 1000 * d2o_margin, he3_ship_usd_kg = ship_he,
                          d2o_ship_usd_kg = ship_d2o, d2o_kWh_kg = e_d2o, station_usd_kWh = usd_kWh))
}

rq3_model <- function(d, mode) {
  sc <- scale_transport(d, d$venus_transport_mult)
  d$tether_fee_usd_kg <- sc$tether; d$station_capex_usd_kg <- sc$station
  if (mode == "protein") return(rq3_protein(d))
  if (mode == "isotope") return(rq3_isotope(d))
  k <- match(d$crop, PRODUCTS$product)
  pr <- PRODUCTS[k, ]
  is_can <- pr$class == "cannabis"
  price <- d$product_price_usd_kg * ifelse(is_can, d$space_premium, 1)
  biomass <- 10^(log10(pr$bio_lo) + d$biomass_u * (log10(pr$bio_hi) - log10(pr$bio_lo)))
  produce <- biomass * d$biomass_cost_usd_kg + ifelse(pr$extract, d$extraction_usd_kg, 0)
  comply <- ifelse(is_can, d$cannabis_compliance_frac * price, 0)

  # station electricity from Venus array physics (daylight H2 production)
  pw <- power_model(d, "section")
  usd_kWh <- pw$venus_isru * d$transport_base_usd_kg * d$venus_transport_mult * 1000 / (8766 * d$life_yr)
  dv <- d$ascent_dv_km_s; f <- d$dry_frac; tfee <- 0
  if (mode == "full") {
    dv <- dv - d$rockoon_save_km_s - d$ramjet_v_km_s - d$tether_tip_km_s
    f <- f + d$wing_frac; tfee <- d$tether_fee_usd_kg
  }
  dv <- pmax(dv, 0.5)
  pf <- stage_pf(dv * 1000, d$isp_s, f)                  # single stage NTR-H2 to LVO
  v_entry <- sqrt(d$venus_v_inf_km_s^2 + 11.18^2)
  epf <- capsule_frac(d, v_entry)                          # aerocapture into Earth orbit
  m0 <- 1 / (pf * epf)
  dry <- f * m0 + (1 / epf - 1)
  prop <- m0 - dry - 1
  hw_per_flight <- dry * (
      d$imported_frac_dry * d$hardware_usd_kg / d$vehicle_reuse +
      d$reactor_frac_dry  * d$hardware_usd_kg * (1 + d$fuel_supply_premium) / pmin(d$reactor_life_flights, d$vehicle_reuse) +
      (1 - d$imported_frac_dry - d$reactor_frac_dry) * d$local_mfg_usd_kg / d$vehicle_reuse)
  depart <- departure_cost(d, MU_VENUS, R_LVO, d$venus_v_inf_km_s, SUN_VENUS)
  launch <- prop * d$prop_kWh_kg * usd_kWh + hw_per_flight +
            d$downmass_usd_kg + depart + tfee
  feasible <- is.finite(m0) & pf > 0
  launch[!feasible] <- CFG$infeasible_cost_usd_kg
  cost <- pmin(produce + comply + launch, CFG$infeasible_cost_usd_kg)
  list(D = price - cost, A = price, B = cost,
       D_lab = "Profit per kg exported (USD/kg)",
       extra = data.frame(product = d$crop, production_usd_kg = produce, launch_usd_kg = launch,
                          compliance_usd_kg = comply, liftoff_kg_per_kg = ifelse(feasible, m0, NA),
                          departure_usd_kg = depart, shuttle_hw_usd_kg = hw_per_flight, station_usd_kWh = usd_kWh,
                          rocket_dv_km_s = dv, feasible = feasible))
}

rq4_model <- function(d, mode) {
  scV <- scale_transport(d, d$venus_transport_mult); scM <- scale_transport(d, d$mars_transport_mult)
  d$station_capex_usd_kg <- scV$station
  pw <- power_model(d, mode)
  r <- d$discount_rate
  ann <- (1 - (1 + r)^(-d$horizon_yr)) / r
  tcV <- d$transport_base_usd_kg * d$venus_transport_mult
  tcM <- d$transport_base_usd_kg * d$mars_transport_mult
  # colony power: cheaper of solar + storage or solar + fission for the night/base share
  fis <- function(tc, extra_mass) d$fission_kg_per_kWe / 1000 * extra_mass * tc +
                                  d$fission_usd_per_kWe / 1000 * (1 + d$fuel_supply_premium)
  fis_norm <- d$life_yr / d$fission_life_yr                   # replacement frequency vs other equipment
  solarV <- pw$venus_colony * tcV
  solarM <- pw$mars_colony  * tcM
  hybV <- (1 - d$ls_frac) * pw$venus_arr * tcV +
          d$ls_frac * fis(tcV, d$venus_fission_shield_mult * d$venus_lift_penalty) * fis_norm
  hybM <- (1 - d$ls_frac) * pw$mars_arr * tcM + d$ls_frac * fis(tcM, 1) * fis_norm
  cpwV <- pmin(solarV, hybV); cpwM <- pmin(solarM, hybM)        # USD per W of colony load
  kWhV <- cpwV * 1000 / (8766 * d$life_yr)
  kWhM <- cpwM * 1000 / (8766 * d$life_yr)

  # thermal control (kW electric per colonist)
  T_in <- 295.15
  Tv <- venus_T_K(d$venus_alt_km)
  intern <- d$internal_heat_frac * d$colony_load_kW
  Qc <- d$k_venus_kW_K * pmax(Tv - T_in, 0) + ifelse(Tv > T_in, intern, 0)
  cop_c <- d$carnot_frac * (T_in - 10) / pmax(Tv + 15 - (T_in - 10), 5)
  elecV <- Qc / pmin(cop_c, 15)
  # heating when the float altitude is colder than the cabin (same method as Mars)
  QhV <- pmax(d$k_venus_kW_K * (T_in - Tv) - intern, 0)
  cop_hV <- d$carnot_frac * (T_in + 10) / pmax((T_in + 10) - (Tv - 15), 5)
  elecV <- elecV + QhV / pmax(cop_hV, 1)
  if (mode == "city") {                         # habitat cooled while at draw depth (0.5 x duty of the year)
    Td <- venus_T_K(d$draw_alt_km)
    Qd <- d$k_venus_kW_K * pmax(Td - T_in, 0) + intern
    cop_d <- d$carnot_frac * (T_in - 10) / pmax(Td + 15 - (T_in - 10), 5)
    elecV <- elecV * (1 - 0.5 * d$draw_duty) + (Qd / pmin(cop_d, 15)) * 0.5 * d$draw_duty
  }
  Tm <- d$mars_ambient_C + 273.15
  Qh <- pmax(d$k_mars_kW_K * (T_in - Tm) - intern, 0)
  cop_h <- d$carnot_frac * (T_in + 10) / ((T_in + 10) - (Tm - 15))
  elecM <- Qh / pmax(cop_h, 1)

  # hydroponic lighting (kW per colonist) and crop CO2 compression
  par <- 0.45 * 4.57 * 0.0864                                # avg W/m2 -> mol/m2/day PAR
  venus_avg_Wm2 <- VENUS_TOA_WM2 * d$venus_cloud_trans * 0.5 * (2 / pi) * cos(d$drift_lat * deg)
  mars_avg_Wm2  <- d$mars_toa_Wm2 * d$mars_tau_trans * d$mars_storm_factor *
                   (1 / pi) * cos(d$mars_site_lat * deg)
  dliV <- venus_avg_Wm2 * d$glazing_trans * par
  dliM <- mars_avg_Wm2 * d$glazing_trans * d$mars_waterwall_par * par
  areaV <- d$crop_area_m2 * d$venus_subst_frac
  areaM <- d$crop_area_m2 * d$mars_subst_frac
  led_kW <- function(mol_day, area) mol_day * area / d$led_umol_J / 86.4   # mol/day -> kW
  lightV <- led_kW(pmax(d$target_DLI - dliV, 0) + d$venus_night_light_frac * 0.5 * d$target_DLI * 0.25, areaV)
  lightM <- led_kW(pmax(d$target_DLI - dliM, 0), areaM)
  co2M_kW <- d$biomass_dry_kg_yr * d$mars_subst_frac * d$co2_kg_per_kg_dry * d$mars_co2_kWh_kg / 8766

  # colony power system capex and replacement
  PV_W <- (d$colony_load_kW + elecV + lightV) * 1000
  PM_W <- (d$colony_load_kW + elecM + lightM + co2M_kW) * 1000
  powV <- PV_W * cpwV
  powM <- PM_W * cpwM

  # gravity: crop yield scales local food; lost hours and attrition are costs
  hm <- health_model(d)
  # protein product mass replacing imports, capped by what crops leave unmet
  ferm_kg <- d$protein_kg_yr * d$ferm_share / d$protein_frac
  leftV <- d$food_import_kg_yr * (1 - pmin(d$venus_subst_frac * hm$cropV, 1))
  leftM <- d$food_import_kg_yr * (1 - pmin(d$mars_subst_frac  * hm$cropM, 1))
  fermV <- pmin(ferm_kg, leftV); fermM <- pmin(ferm_kg, leftM)
  savV <- (d$food_import_kg_yr - leftV + fermV) * tcV
  savM <- (d$food_import_kg_yr - leftM + fermM) * tcM
  e_ferm <- function(kg) kg * d$protein_frac * d$protein_kWh_kg                       # kWh/yr
  protV <- e_ferm(fermV) * kWhV + fermV * d$protein_frac * 0.16 * d$venus_N_kWh_kg * kWhV
  protM <- e_ferm(fermM) * kWhM + fermM * d$protein_frac * 0.16 * pmin(d$mars_N_kWh_kg * kWhM, tcM) +
           fermM * 1.8 * d$mars_co2_kWh_kg * kWhM                                     # CO2 from 6 mbar
  fermCapV <- fermV * d$ferm_kg_per_kgyr * tcV
  fermCapM <- fermM * d$ferm_kg_per_kgyr * tcM
  gravV <- hm$lostV * d$labor_value_usd_hr + hm$attrV * d$replace_usd
  gravM <- hm$lostM * d$labor_value_usd_hr + hm$attrM * d$replace_usd

  # fertilizer make-up (USD/yr)
  mk <- function(sub) d$biomass_dry_kg_yr * sub * (1 - d$nutrient_recycle)
  bV <- mk(d$venus_subst_frac); bM <- mk(d$mars_subst_frac)
  fertV <- bV * NUTRIENT["N"] * d$venus_N_kWh_kg * kWhV +                  # S local, acid byproduct
           bV * (NUTRIENT["P"] + NUTRIENT["K"] + NUTRIENT["other"]) *
           pmin(d$venus_surface_mineral_usd_kg, tcV)                         # surface-mined or imported
  fertM <- bM * NUTRIENT["N"] * pmin(d$mars_N_kWh_kg * kWhM, tcM) +
           bM * (NUTRIENT["P"] + NUTRIENT["K"]) * pmin(d$mars_PK_kWh_kg * kWhM, tcM) +
           bM * (NUTRIENT["S"] + NUTRIENT["other"]) * pmin(d$mars_S_other_kWh_kg * kWhM, tcM)

  # exports
  delayV <- 0.5 * SYNODIC_VENUS_D / 365.25 + d$venus_transit_yr
  delayM <- 0.5 * SYNODIC_MARS_D  / 365.25 + d$mars_transit_yr
  shuttle <- function(dv, kWh_prop, usd_kWh, extra_f = 0) {  # USD per kg delivered to low orbit
    f <- d$ntr_dry_frac + extra_f
    pf <- stage_pf(pmax(dv, 0.5) * 1000, d$ntr_isp_s, f)
    m0 <- 1 / pf; dry <- f * m0; prop <- m0 - dry - 1
    hw <- dry * (d$imported_frac_dry * d$ntr_hw_usd_kg / d$ntr_reuse +
                 d$reactor_frac_dry * d$ntr_hw_usd_kg * (1 + d$fuel_supply_premium) / pmin(d$reactor_life_flights, d$ntr_reuse) +
                 (1 - d$imported_frac_dry - d$reactor_frac_dry) * d$local_mfg_usd_kg / d$ntr_reuse)
    out <- prop * kWh_prop * usd_kWh + hw
    out[!is.finite(out) | pf <= 0] <- CFG$infeasible_cost_usd_kg
    out
  }
  expCostV <- shuttle(d$venus_ascent_dv_km_s - d$venus_rockoon_save_km_s - d$venus_ramjet_v_km_s - d$tether_tip_km_s,
                      d$venus_h2_kWh_kg, kWhV, d$venus_wing_frac) +
              departure_cost(d, MU_VENUS, R_LVO, d$venus_v_inf_km_s, SUN_VENUS) + scV$tether
  expCostM <- shuttle(d$mars_ascent_dv_km_s - d$tether_tip_km_s,
                      d$mars_h2_kWh_kg, kWhM) +
              departure_cost(d, MU_MARS, R_LMO, d$mars_v_inf_km_s, SUN_MARS) + scM$tether
  expV <- d$export_kg_yr * (d$export_value_usd_kg * (1 + r)^(-delayV) - expCostV)
  expM <- d$export_kg_yr * (d$export_value_usd_kg * (1 + r)^(-delayM) - expCostM)

  # isotope co-products (He-3 Venus only; D2O both planets), each line run only if profitable
  wy <- d$isru_water_kg_yr
  airV <- wy * d$venus_air_kg_per_kg_water
  he3V <- airV * HE3_MASS_FRAC(d$venus_He_ppm, d$he3_ratio) * d$he_recovery
  heV  <- airV * HE_MASS_FRAC(d$venus_He_ppm) * d$he_recovery
  # market depth: colony supply vs the Venus share of a small market, constant elasticity
  Mv <- d$he3_market_kg_yr * (1 - d$moon_share) / d$colony_size_n      # market per colonist
  q3 <- ifelse(d$he3_elasticity < 1, pmin(he3V, Mv), he3V)              # sales per colonist
  frac3 <- ifelse(he3V > 0, q3 / he3V, 0)                               # share of air processed
  price3 <- d$he3_price_usd_kg * pmax(q3 / Mv, 1)^(-1 / d$he3_elasticity)
  he3NetV <- q3 * (price3 * (1 + r)^(-delayV) - expCostV) -
             frac3 * (airV * d$he_kWh_kg_air + heV * d$he4_kWh_kg) * kWhV * (1 + d$he_capex_mult)
  d2oLine <- function(DH, kWh, cost_kg) {
    kg <- wy * D2O_PER_KG_WATER(DH) * d$d2o_recovery
    kg * (d$d2o_price_usd_kg + d$earth_launch_usd_kg - cost_kg) -
      kg * d$d2o_ref_kWh_kg * (VSMOW_DH / DH) * (1 - d$d2o_synergy) * kWh * (1 + d$d2o_capex_mult)
  }
  d2oNetV <- d2oLine(d$venus_DH, kWhV, expCostV)
  d2oNetM <- d2oLine(d$mars_DH, kWhM, expCostM)
  isoV <- pmax(he3NetV, 0) + pmax(d2oNetV, 0)
  isoM <- pmax(d2oNetM, 0)

  # bulk supply to cislunar buyers at Earth-launch parity (Venus N2, Mars water)
  supV <- pmax(d$supply_kg_yr * (d$earth_launch_usd_kg -
               d$venus_N2_kWh_kg * kWhV * (1 + d$supply_capex_mult) - expCostV), 0)
  supM <- pmax(d$supply_kg_yr * (d$earth_launch_usd_kg - d$mars_water_usd_kg - expCostM), 0)
  # quality-of-life hardship premium on colonist compensation (gravity excluded)
  qolV <- d$colonist_comp_usd_yr * d$venus_hardship_frac
  qolM <- d$colonist_comp_usd_yr * d$mars_hardship_frac

  wall_kgV <- d$venus_shelter_area_m2 * d$venus_shelter_g_cm2 * 10
  wall_kgM <- d$mars_wall_area_m2 * d$mars_wall_g_cm2 * 10
  wallV <- pmax(wall_kgV - d$water_reserve_kg, 0) * d$venus_water_usd_kg * d$venus_lift_penalty
  wallM <- pmax(wall_kgM - d$water_reserve_kg, 0) * d$mars_water_usd_kg
  # greenhouse structure from pressure physics (kg per colonist)
  memb <- function(dp_kPa) dp_kPa * 1000 * d$gh_radius_m * d$gh_SF / (d$spec_strength_kJ_kg * 1000)
  agV <- areaV * (d$hydro_kg_m2 + d$env_area_factor * d$frame_mult * memb(d$venus_gh_dp_kPa))
  agM <- areaM * (d$hydro_kg_m2 + d$env_area_factor * d$frame_mult * memb(d$mars_gh_kPa - 0.6))
  ledV <- lightV * d$led_kg_per_kW * tcV
  ledM <- lightM * d$led_kg_per_kW * tcM
  capV <- agV * tcV + d$venus_launch_capex_usd + powV + wallV + ledV + fermCapV
  # cloud-level weather: gust/altitude-control margin and thinner-air envelope penalty
  gustV <- d$venus_gust_capex_frac * ifelse(d$venus_alt_km < d$conv_top_km, 1, 0.3)
  envV  <- d$envelope_capex_frac * pmax(venus_rho(52.5) / venus_rho(d$venus_alt_km) - 1, 0)
  capV  <- capV * (1 + gustV + envV)
  capM <- agM * tcM + d$mars_launch_capex_usd  + powM + wallM + ledM + fermCapM
  resV <- d$research_usd_yr * d$venus_science_mult / d$colony_size_n
  resM <- d$research_usd_yr * d$mars_science_mult / d$colony_size_n
  govV <- d$gov_cost_share * capV * (isoV > 0)
  govM <- d$gov_cost_share * capM * (isoM > 0)
  npvV <- -capV + govV + ann * (savV + expV + isoV + supV + resV - qolV - fertV - d$om_frac * capV - powV / d$life_yr - gravV - protV)
  npvM <- -capM + govM + ann * (savM + expM + isoM + supM + resM - qolM - fertM - d$om_frac * capM - powM / d$life_yr - gravM - protM)
  list(D = npvV - npvM, A = npvV, B = npvM,
       D_lab = "Venus minus Mars NPV (USD/colonist)",
       extra = data.frame(venus_thermal_kWe = elecV, mars_thermal_kWe = elecM,
                          venus_power_capex = powV, mars_power_capex = powM,
                          venus_fert_usd_yr = fertV, mars_fert_usd_yr = fertM,
                          venus_usd_kWh = kWhV, mars_usd_kWh = kWhM,
                          venus_export_cost = expCostV, mars_export_cost = expCostM,
                          venus_waterwall_usd = wallV, mars_waterwall_usd = wallM,
                          venus_natural_DLI = dliV, mars_natural_DLI = dliM,
                          venus_led_kW = lightV, mars_led_kW = lightM, mars_crop_co2_kW = co2M_kW,
                          venus_gravity_cost_yr = gravV, mars_gravity_cost_yr = gravM,
                          venus_ag_kg = agV, mars_ag_kg = agM,
                          venus_protein_usd_yr = protV, mars_protein_usd_yr = protM,
                          venus_he3_net_usd_yr = he3NetV, venus_d2o_net_usd_yr = d2oNetV, mars_d2o_net_usd_yr = d2oNetM,
                          venus_isotope_usd_yr = isoV, mars_isotope_usd_yr = isoM,
                          venus_supply_usd_yr = supV, mars_supply_usd_yr = supM,
                          venus_he3_sold_kg_yr = q3, venus_he3_price = price3,
                          venus_gov_share_usd = govV, mars_gov_share_usd = govM,
                          venus_research_usd_yr = resV, mars_research_usd_yr = resM,
                          venus_weather_capex_frac = gustV + envV, venus_heating_kWe = QhV / pmax(cop_hV, 1), venus_hab_T_C = Tv - 273.15,
                          traffic_t_yr = scV$traffic_t_yr, venus_tether_usd_kg = scV$tether, mars_tether_usd_kg = scM$tether, station_usd_kg = scV$station,
                          venus_hardship_usd_yr = qolV, mars_hardship_usd_yr = qolM,
                          venus_capex_per_colonist = capV, mars_capex_per_colonist = capM,
                          venus_claim_pv = ann * (isoV + supV + pmax(expV, 0)), mars_claim_pv = ann * (isoM + supM + pmax(expM, 0)),
                          venus_cut_to_3yr_pay = capV / (3 * d$colonist_comp_usd_yr),
                          mars_cut_to_3yr_pay = capM / (3 * d$colonist_comp_usd_yr),
                          venus_fission_chosen = hybV < solarV, mars_fission_chosen = hybM < solarM,
                          venus_usd_per_W = cpwV, mars_usd_per_W = cpwM, venus_night_h = pw$venus_night_h))
}

rq5_model <- function(d, mode) {
  hm <- health_model(d)
  margin <- CFG$rq5_margin_frac * d$work_hours_yr
  list(D = hm$hrsV - hm$hrsM - margin, A = hm$hrsV, B = hm$hrsM,
       D_lab = "Venus minus Mars productive hours, net of 5% margin (h/colonist-yr)",
       extra = data.frame(venus_deficit = hm$defV, mars_deficit = hm$defM,
                          venus_lost_h = hm$lostV, mars_lost_h = hm$lostM,
                          gap_h = hm$hrsV - hm$hrsM, margin_h = margin,
                          mars_crop_yield = hm$cropM, venus_crop_yield = hm$cropV))
}

RQ_DEF <- list(
  RQ1 = list(model = rq1_model, transform = "asinh",
    H0 = "The modeled annual dose equivalent for a Venus aerostat habitat is not lower than that for a Mars surface habitat."),
  RQ2 = list(model = rq2_model, transform = "asinh",
    H0 = "The energy and mass cost per kilogram of delivered H2O and O2 from Venus atmospheric extraction is not lower than that from Mars ice mining and purification."),
  RQ3 = list(model = rq3_model, transform = "asinh",
    H0 = "The landed value of Venus-produced high-value botanical pharmaceutical and cannabis products does not exceed their full delivered cost per kilogram to Earth (rocket, full); for the protein scenario, the delivered cost of Venus microbial protein to off-Earth markets is not lower than supply from Earth; for the isotope scenario, the net margin from helium-3 and heavy-water co-products of sulfuric acid mining does not exceed zero."),
  RQ4 = list(model = rq4_model, transform = "asinh",
    H0 = "The net present value of a Venus colony economy is not greater than that of a comparable Mars colony economy."),
  RQ5 = list(model = rq5_model, transform = "none",
    H0 = "The Venus advantage in health-adjusted productive hours per colonist-year does not exceed 5% of scheduled working hours.")
)

# H2O helpers
# -----------------------------------------------------------------------------
# Shut down any H2O cluster left running by a previous session, so this run
# starts clean with its own nthreads / max_mem_size and an empty object store.
h2o_port <- 54321
h2o_probe <- function() {
  tryCatch(h2o.clusterIsUp(conn = new("H2OConnection", ip = "localhost", port = h2o_port)),
           error = function(e) FALSE)
}
if (h2o_probe()) {
  message("An H2O cluster from a previous session is running on port ", h2o_port,
          "; shutting it down before starting this run.")
  try(suppressWarnings(h2o.connect(ip = "localhost", port = h2o_port)), silent = TRUE)
  try(h2o.shutdown(prompt = FALSE), silent = TRUE)
  t0 <- Sys.time()
  while (h2o_probe() && difftime(Sys.time(), t0, units = "secs") < 60) Sys.sleep(2)
  if (h2o_probe()) stop("The previous H2O cluster on port ", h2o_port,
                        " did not shut down within 60 seconds. Close the other R session or end its java.exe process, then rerun this script.")
  Sys.sleep(3)   # let the JVM release the port fully (slower on Windows)
  message("Previous H2O cluster shut down.")
}
# Raise the H2O client's curl connect timeout from the curl default (10 s) to 120 s
local({
  f <- get(".h2o.doRawREST.with.curl", envir = asNamespace("h2o"))
  b <- body(f)
  bl <- as.list(b)
  bl[[2]] <- quote(timeout_secs <- getOption("vm_h2o_request_timeout", 900))   # was 0 (no limit)
  b <- as.call(append(bl, quote(curl::handle_setopt(handle, connecttimeout = 120L)), after = 3))
  body(f) <- b
  utils::assignInNamespace(".h2o.doRawREST.with.curl", f, ns = "h2o")
})

# PID(s) of the process listening on the H2O port
h2o_port_pids <- function(port = h2o_port) {
  pid <- integer(0)
  if (.Platform$OS.type == "windows") {
    out <- tryCatch(system2("netstat", c("-ano", "-p", "TCP"), stdout = TRUE, stderr = FALSE),
                    error = function(e) character(0))
    ln <- grep(sprintf("[:.]%d\\s.*LISTENING", port), out, value = TRUE)
    pid <- suppressWarnings(as.integer(sub(".*\\s(\\d+)\\s*$", "\\1", ln)))
  } else {
    out <- tryCatch(suppressWarnings(system2("lsof", c("-t", sprintf("-iTCP:%d", port), "-sTCP:LISTEN"),
                                              stdout = TRUE, stderr = FALSE)), error = function(e) character(0))
    pid <- suppressWarnings(as.integer(out))
  }
  unique(pid[!is.na(pid) & pid > 0])
}
h2o_start <- function() {
  log_dir <- file.path(CFG$out_dir, "h2o_logs")
  dir.create(log_dir, showWarnings = FALSE, recursive = TRUE)
  h2o.init(nthreads = CFG$nthreads, max_mem_size = CFG$max_mem, log_dir = normalizePath(log_dir), log_level = "INFO")
  h2o.no_progress()
}
h2o_up <- function() tryCatch(isTRUE(h2o.clusterIsUp()), error = function(e) FALSE)
h2o_restart <- function(reason = "") {
  message("Restarting H2O cluster", if (nzchar(reason)) paste0(" (", reason, ")"), " ...")
  try(h2o.shutdown(prompt = FALSE), silent = TRUE)
  t0 <- Sys.time()
  while (length(h2o_port_pids()) > 0 && difftime(Sys.time(), t0, units = "secs") < 30) Sys.sleep(2)
  pids <- h2o_port_pids()
  if (length(pids) > 0) {                      # hung JVM: end the process holding the port
    message("H2O did not shut down; ending process ", paste(pids, collapse = ", "))
    for (pid in pids) try(tools::pskill(pid, tools::SIGKILL), silent = TRUE)
    t1 <- Sys.time()
    while (length(h2o_port_pids()) > 0 && difftime(Sys.time(), t1, units = "secs") < 30) Sys.sleep(2)
  }
  Sys.sleep(5)
  h2o_start()
}
dir.create(CFG$out_dir, showWarnings = FALSE, recursive = TRUE)
h2o_start()

# Retry a single H2O operation in place after a transient client-side failure
h2o_err_log <- function(msg) {
  line <- sprintf("%s  %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg)
  message(line)
  try(cat(line, "\n", file = file.path(CFG$out_dir, "h2o_errors.log"), append = TRUE), silent = TRUE)
}
h2o_retry <- function(expr, what = "H2O call", tries = 6) {
  ex <- substitute(expr); env <- parent.frame()
  for (i in seq_len(tries)) {
    res <- tryCatch({
      if (getOption("vm_fault_rate", 0) > runif(1)) stop("injected test fault")
      list(ok = TRUE, value = eval(ex, env))
    }, error = function(e) list(ok = FALSE, msg = conditionMessage(e)))
    if (res$ok) return(res$value)
    wait <- min(10 * i, 60)
    h2o_err_log(sprintf("%s failed (try %d/%d): %s; retrying in %d s",
                        what, i, tries, gsub("[\r\n]+", " ", res$msg), wait))
    Sys.sleep(getOption("vm_retry_wait_scale", 1) * wait)
    if (!h2o_up()) {
      Sys.sleep(getOption("vm_retry_wait_scale", 1) * 30)
      if (!h2o_up()) stop(sprintf("H2O is down during %s: %s", what, res$msg))
    }
  }
  stop(sprintf("%s failed %d times: %s", what, tries, res$msg))
}

fit_ann <- function(df, xcols, seed) {
  h2o_retry({
  fr <- as.h2o(df)
  m <- tryCatch(h2o.deeplearning(x = xcols, y = "y", training_frame = fr,
                        hidden = CFG$hidden, activation = CFG$activation,
                        categorical_encoding = CFG$cat_encoding, use_all_factor_levels = TRUE,
                        hidden_dropout_ratios = CFG$hidden_dropout,
                        epochs = CFG$epochs, l2 = CFG$l2, seed = seed,
                        standardize = TRUE, stopping_rounds = 0, verbose = FALSE),
                finally = try(h2o.rm(fr), silent = TRUE))
  m
  }, what = "fit")
}
pred_vec <- function(m, fr) h2o_retry({
  pf <- h2o.predict(m, fr)
  out <- as.data.frame(pf)$predict
  try(h2o.rm(pf), silent = TRUE)               # do not leave prediction frames in the cluster
  out
}, what = "predict")
rmse <- function(a, b) sqrt(mean((a - b)^2))

# -----------------------------------------------------------------------------
# Per-RQ pipeline
# -----------------------------------------------------------------------------
run_rq <- function(rq, mode) {
  t0 <- Sys.time()
  lab <- if (mode %in% c("city", "section", "rocket", "full", "protein", "isotope")) paste(rq, mode, sep = "_") else rq
  cat(sprintf("\n================ %s ================\n", lab))
  def <- RQ_DEF[[rq]]
  tab <- PARAMS[PARAMS$rq == rq & PARAMS$mode %in% c("both", mode, if (mode %in% c("rocket", "full")) "export"), ]
  rq_seed <- CFG$seed + match(rq, c("RQ1", "RQ2", "RQ3", "RQ4", "RQ5")) * 100000
  set.seed(rq_seed)

  # ---- Monte Carlo draws and the primary hypothesis test ----
  d <- draw_params(tab, CFG$n_mc)
  res <- def$model(d, mode)
  D <- res$D
  p_mc <- (1 + sum(D <= 0)) / (CFG$n_mc + 1)
  q <- quantile(D, c(0.025, 0.05, 0.5, 0.975), names = FALSE)
  decision <- ifelse(p_mc < CFG$alpha, "Reject H0", "Fail to reject H0")
  cat(sprintf("p_MC = %.4g  median D = %.4g  5th pct = %.4g  -> %s\n",
              p_mc, q[3], q[2], decision))

  draws_out <- cbind(d, D = D, arm_A = res$A, arm_B = res$B)
  if (!is.null(res$extra)) draws_out <- cbind(draws_out, res$extra)
  write.csv(draws_out, file.path(CFG$out_dir, paste0(lab, "_mc_draws.csv")), row.names = FALSE)

  # ---- response transform for the ANN ----
  s <- median(abs(D)); if (!is.finite(s) || s == 0) s <- 1
  ytf <- if (def$transform == "asinh") function(v) asinh(v / s) else identity
  X <- to_features(d, tab)
  xcols <- names(X)
  is_fac <- sapply(X, is.factor)
  dat <- cbind(X, y = ytf(D))

  idx <- seq_len(CFG$n_mc)
  i_tr <- idx[1:CFG$n_train]
  i_ho <- idx[CFG$n_train + 1:CFG$n_holdout]
  i_ev <- idx[CFG$n_train + CFG$n_holdout + 1:CFG$n_eval]
  train <- dat[i_tr, ]; hold <- dat[i_ho, ]; ev <- dat[i_ev, ]

  # Perturbation stack for local sensitivities (uploaded once)
  sds <- sapply(train[, xcols, drop = FALSE], function(v) if (is.factor(v)) NA else sd(v))
  h <- CFG$perturb_sd
  stack <- ev[, xcols, drop = FALSE]
  blocks <- list()                                   # row blocks after the base rows
  for (j in xcols) {
    if (is_fac[j]) {
      for (lv in levels(ev[[j]])) {
        alt <- ev[, xcols, drop = FALSE]; alt[[j]] <- factor(lv, levels = levels(ev[[j]]))
        stack <- rbind(stack, alt); blocks[[length(blocks) + 1]] <- list(col = j, kind = "level", lv = lv)
      }
    } else {
      up <- ev[, xcols, drop = FALSE]; up[[j]] <- up[[j]] + h * sds[j]
      dn <- ev[, xcols, drop = FALSE]; dn[[j]] <- dn[[j]] - h * sds[j]
      stack <- rbind(stack, up, dn)
      blocks[[length(blocks) + 1]] <- list(col = j, kind = "up")
      blocks[[length(blocks) + 1]] <- list(col = j, kind = "dn")
    }
  }
  fr_stack <- h2o_retry(as.h2o(stack), what = "upload stack")
  fr_hold  <- h2o_retry(as.h2o(hold[, xcols, drop = FALSE]), what = "upload holdout")
  ne <- nrow(ev); P <- length(xcols)

  sens  <- array(NA_real_, c(CFG$B, ne, P), dimnames = list(NULL, NULL, xcols))
  mean_pred <- r2_hold <- rmse_int <- numeric(CFG$B)
  rmse_perm <- matrix(NA_real_, CFG$B_null, P, dimnames = list(NULL, xcols))

  for (b in seq_len(CFG$B)) {
    bi <- sample(nrow(train), replace = TRUE)
    tb <- train[bi, ]
    m <- fit_ann(tb, xcols, seed = rq_seed + b)
    pr <- pred_vec(m, fr_stack)
    base <- pr[1:ne]
    blk <- function(i) pr[ne * i + 1:ne]
    for (k in seq_len(P)) {
      j <- xcols[k]
      ids <- which(sapply(blocks, function(z) z$col == j))
      if (is_fac[j]) {                         # counterfactual: mean |change| over other levels
        own <- as.character(ev[[j]])
        dif <- sapply(ids, function(i) ifelse(own == blocks[[i]]$lv, NA, abs(blk(i) - base)))
        sens[b, , k] <- rowMeans(matrix(dif, nrow = ne), na.rm = TRUE)
      } else {
        sens[b, , k] <- (blk(ids[1]) - blk(ids[2])) / (2 * h)   # d(y) per 1 SD of input k
      }
    }
    mean_pred[b] <- mean(base)
    ph <- pred_vec(m, fr_hold)
    rmse_int[b] <- rmse(ph, hold$y)
    r2_hold[b] <- 1 - sum((ph - hold$y)^2) / sum((hold$y - mean(hold$y))^2)
    try(h2o.rm(m), silent = TRUE)

    # paired permutation null arm on the same bootstrap sample
    if (b <= CFG$B_null) {
      for (k in seq_len(P)) {
        tp <- tb; tp[[xcols[k]]] <- sample(tp[[xcols[k]]])
        mp <- fit_ann(tp, xcols, seed = rq_seed + b)
        rmse_perm[b, k] <- rmse(pred_vec(mp, fr_hold), hold$y)
        try(h2o.rm(mp), silent = TRUE)
      }
    }
    if (b %% 10 == 0 || b == CFG$B)
      cat(sprintf("  %s bootstrap %d/%d  holdout R2 = %.3f  elapsed %.1f min\n",
                  lab, b, CFG$B, r2_hold[b], as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  }
  try(h2o.rm(fr_stack), silent = TRUE); try(h2o.rm(fr_hold), silent = TRUE)

  # ---- variable-level results ----
  glob <- apply(abs(sens), c(1, 3), mean)            # B x P
  signed <- apply(sens, c(1, 3), mean)
  nb <- CFG$B_null
  p_null <- sapply(seq_len(P), function(k)
    (1 + sum(rmse_perm[1:nb, k] <= rmse_int[1:nb])) / (nb + 1))
  d_rmse <- rmse_perm[1:nb, , drop = FALSE] - rmse_int[1:nb]
  vt <- data.frame(
    variable = xcols,
    units = ifelse(is_fac[xcols], "factor", tab$units[match(xcols, tab$name)]),
    mean_abs_sensitivity = colMeans(glob),
    abs_sens_ci_lo = apply(glob, 2, quantile, 0.025),
    abs_sens_ci_hi = apply(glob, 2, quantile, 0.975),
    mean_signed_sensitivity = colMeans(signed),
    signed_ci_lo = apply(signed, 2, quantile, 0.025),
    signed_ci_hi = apply(signed, 2, quantile, 0.975),
    delta_rmse_permuted = colMeans(d_rmse),
    delta_rmse_ci_lo = apply(d_rmse, 2, quantile, 0.025),
    delta_rmse_ci_hi = apply(d_rmse, 2, quantile, 0.975),
    p_null_arm = p_null,
    significant = p_null < CFG$alpha,
    source = ifelse(is_fac[xcols], FACTOR_INFO[xcols], tab$source[match(xcols, tab$name)]),
    stringsAsFactors = FALSE)
  vt <- vt[order(-vt$mean_abs_sensitivity), ]
  vt$rank <- seq_len(nrow(vt))
  write.csv(vt, file.path(CFG$out_dir, paste0(lab, "_variable_tests.csv")), row.names = FALSE)

  # ---- per-prediction (local) sensitivities ----
  loc_mean <- apply(sens, c(2, 3), mean)
  loc_lo <- apply(sens, c(2, 3), quantile, 0.025)
  loc_hi <- apply(sens, c(2, 3), quantile, 0.975)
  D_ev <- D[i_ev]
  regime <- factor(ifelse(D_ev > 0, "favors_H1", "favors_H0"))
  regime_basis <- "sign of D (D > 0 vs D <= 0)"
  if (min(table(factor(regime, levels = c("favors_H0", "favors_H1")))) < 5) {
    # All draws on one side of zero: group by outcome magnitude instead
    regime <- factor(ifelse(D_ev > median(D_ev), "upper_half_D", "lower_half_D"))
    regime_basis <- "median split of D (all evaluation draws share one sign)"
  }
  local_out <- data.frame(eval_row = i_ev, D = D_ev, regime = regime,
                          setNames(as.data.frame(loc_mean), paste0("sens_", xcols)),
                          setNames(as.data.frame(loc_lo), paste0("lo_", xcols)),
                          setNames(as.data.frame(loc_hi), paste0("hi_", xcols)))
  if (!is.null(d$crop)) local_out$crop <- d$crop[i_ev]
  write.csv(local_out, file.path(CFG$out_dir, paste0(lab, "_local_sensitivity.csv")), row.names = FALSE)

  # ---- PERMANOVA on the local sensitivity matrix ----
  perm <- list(F = NA, R2 = NA, p = NA, note = "")
  tabreg <- table(regime)
  if (length(tabreg) == 2 && min(tabreg) >= 5) {
    a2 <- adonis2(dist(loc_mean) ~ regime, data = data.frame(regime = regime),
                  permutations = CFG$permanova_perm)
    perm <- list(F = a2$F[1], R2 = a2$R2[1], p = a2$`Pr(>F)`[1],
                 note = paste("Grouping:", regime_basis))
    capture.output(print(a2), file = file.path(CFG$out_dir, paste0(lab, "_adonis2.txt")))
  } else {
    perm$note <- sprintf("PERMANOVA skipped: regime counts %s",
                         paste(names(tabreg), tabreg, collapse = ", "))
    cat("  ", perm$note, "\n")
  }

  # ---- figures (bold strokes for legibility) ----
  png(file.path(CFG$out_dir, paste0(lab, "_D_distribution.png")), 1800, 1200, res = 200)
  par(mar = c(5, 5, 4, 2), lwd = 2.5)
  yplot <- if (def$transform == "asinh") asinh(D / s) else D
  hist(yplot, breaks = 80, col = "grey75", border = "grey30",
       main = sprintf("%s: %s", lab, res$D_lab),
       xlab = if (def$transform == "asinh") sprintf("asinh(D / %.3g)", s) else res$D_lab,
       cex.main = 1.1, cex.lab = 1.2, font.lab = 2)
  abline(v = 0, lwd = 4, col = "black")
  abline(v = if (def$transform == "asinh") asinh(q[2] / s) else q[2], lwd = 4, lty = 2, col = "firebrick")
  legend("topright", c("D = 0", "5th percentile"), lwd = 4, lty = c(1, 2),
         col = c("black", "firebrick"), bty = "n")
  mtext(sprintf("p_MC = %.4g  (%s at alpha = %.2f)", p_mc, decision, CFG$alpha), side = 3, line = 0.2)
  dev.off()

  png(file.path(CFG$out_dir, paste0(lab, "_sensitivity.png")), 1800, 200 + 70 * P, res = 200)
  par(mar = c(5, 16, 3, 2))
  o <- rev(seq_len(nrow(vt)))
  plot(vt$mean_abs_sensitivity[o], seq_along(o), xlim = range(0, vt$abs_sens_ci_hi),
       pch = 19, cex = 1.6, yaxt = "n", ylab = "",
       xlab = "Mean |local sensitivity| (response units per 1 SD of input)",
       main = sprintf("%s: ANN perturbation sensitivities (B = %d)", lab, CFG$B), font.lab = 2)
  segments(vt$abs_sens_ci_lo[o], seq_along(o), vt$abs_sens_ci_hi[o], seq_along(o), lwd = 3)
  axis(2, at = seq_along(o),
       labels = sprintf("%s%s", vt$variable[o], ifelse(vt$significant[o], " *", "")),
       las = 1, cex.axis = 0.75)
  mtext("* null-arm p < alpha", side = 1, line = 4, adj = 1, cex = 0.8)
  dev.off()

  data.frame(
    RQ = rq, venus_mode = mode, run = lab, H0 = def$H0, response = res$D_lab, n_mc = CFG$n_mc,
    median_D = q[3], interval95_lo = q[1], interval95_hi = q[4],
    one_sided_5th_pct = q[2], prob_D_gt_0 = mean(D > 0), p_MC = p_mc,
    alpha = CFG$alpha, decision = decision,
    ann_holdout_R2_median = median(r2_hold),
    ann_mean_pred_ci_lo = quantile(mean_pred, 0.025, names = FALSE),
    ann_mean_pred_ci_hi = quantile(mean_pred, 0.975, names = FALSE),
    n_sig_variables = sum(vt$significant),
    adonis2_F = perm$F, adonis2_R2 = perm$R2, adonis2_p = perm$p,
    permanova_note = perm$note,
    minutes = round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1),
    stringsAsFactors = FALSE)
}


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
n_fits <- sapply(CFG$runs, function(z)
  CFG$B + CFG$B_null * sum(PARAMS$rq == z[1] & PARAMS$mode %in% c("both", z[2])))
cat(sprintf("Total ANN fits planned: %d\n", sum(n_fits)))

write.csv(PARAMS, file.path(CFG$out_dir, "parameter_table.csv"), row.names = FALSE)
write.csv(CROPS, file.path(CFG$out_dir, "crop_price_table.csv"), row.names = FALSE)

# Checkpointed, retrying driver: a lost H2O connection restarts the cluster and
# repeats that run from its start (same seed); finished runs are skipped on rerun.
ckpt_dir <- file.path(CFG$out_dir, "checkpoints")
dir.create(ckpt_dir, showWarnings = FALSE, recursive = TRUE)
run_rq_safe <- function(rq, mode, tries = 3) {
  lab <- if (mode %in% c("city", "section", "rocket", "full", "protein", "isotope")) paste(rq, mode, sep = "_") else rq
  ck <- file.path(ckpt_dir, paste0(lab, "_summary.rds"))
  if (file.exists(ck)) {
    cat(sprintf("\n%s already finished (checkpoint found); skipping.\n", lab))
    return(readRDS(ck))
  }
  h2o_restart(paste("fresh JVM for", lab))
  for (k in seq_len(tries)) {
    out <- tryCatch(run_rq(rq, mode), error = function(e) {
      h2o_err_log(sprintf("%s attempt %d failed: %s", lab, k, gsub("[\r\n]+", " ", conditionMessage(e))))
      NULL
    })
    if (!is.null(out)) {
      saveRDS(out, ck)
      try(h2o.removeAll(), silent = TRUE); invisible(gc())
      return(out)
    }
    invisible(gc())
    h2o_restart(sprintf("after failed attempt %d", k))
  }
  stop(sprintf("%s failed %d times; see %s for the H2O log.", lab, tries, file.path(CFG$out_dir, "h2o_logs")))
}
summary_tab <- do.call(rbind, lapply(CFG$runs, function(z) run_rq_safe(z[1], z[2])))
write.csv(summary_tab, file.path(CFG$out_dir, "hypothesis_summary.csv"), row.names = FALSE)
print(summary_tab[, c("run", "median_D", "one_sided_5th_pct", "p_MC", "decision",
                      "ann_holdout_R2_median", "adonis2_p")])
writeLines(capture.output(sessionInfo()), file.path(CFG$out_dir, "sessionInfo.txt"))
cat("\nDone. Outputs in", normalizePath(CFG$out_dir), "\n")
