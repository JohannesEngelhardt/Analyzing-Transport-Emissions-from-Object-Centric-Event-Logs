![GTFS2OCEL Demo](demo.gif)


# GTFS2OCEL

Welcome to **GTFS2OCEL**, a transformation tool for generating object-centric event logs (OCEL 2.0) from GTFS Schedule and GTFS-Realtime data. The tool retrieves public transport data from the **KODA** platform, preprocesses the GTFS-Realtime snapshots, enriches the data with operational attributes, and transforms the prepared data into sustainability-oriented OCELs for object-centric process mining.

GTFS2OCEL provides the following main features:

* Retrieving GTFS Schedule and GTFS-Realtime data from the KODA platform
* Preprocessing GTFS-Realtime protobuf snapshots into a consolidated tabular representation
* Enriching the prepared data with operational attributes such as occupancy, speed, travel duration, and traveled distance
* Transforming the enriched GTFS data into OCEL 2.0 event logs
* Exporting the generated OCELs for further analysis in object-centric process mining and sustainability applications

The project consists of:

* A **Python** backend implementing the complete data retrieval, preprocessing, enrichment, and GTFS-to-OCEL transformation pipeline
* An **HTML** user interface for configuring the transformation pipeline and generating OCELs
