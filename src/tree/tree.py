import io
import logging
import os
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from constraint import BaseConstraint
from database_node import Base, DatabaseNode
from matplotlib import cm
from node import TreeNode
from pydantic import BaseModel
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from ..martingale.base_martingale_model import BaseMartingaleModel
from ..martingale.copula_model import CopulaModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScenarioTree(BaseModel):
    """
    Scenario Tree model for lazy loading of nodes from a SQLite database.

    This model represents a scenario tree structure where each node can be lazily loaded.
    The tree is represented as a list of tuples, where each tuple contains the node ID and the parent ID.
    """

    # List of tuples representing the scenario tree (node_id, parent_id)
    tree: List[Tuple[int, Optional[int]]] = []

    # Name of the scenario tree
    tree_name: str = "scenario_tree"

    # Database engine for SQLite
    _db_engine: Optional[Engine] = None

    # Model for generating the predictive component of the scenario
    predictive_model: Optional[Any] = None

    # Model for generating the martingale component of the scenario
    martingale_model: Optional[BaseMartingaleModel] = None

    # Model input data for generating scenarios
    model_input: Optional[np.ndarray] = None

    # Scenario values for the martingale process
    scenario_value: Optional[np.ndarray] = None

    # Constraints for scenario tree nodes
    constraints: Optional[List[BaseConstraint]] = None

    # Names of the columns in the model input data
    cols_name: Optional[List[str]] = []

    # Dates corresponding to the model input data
    dates: Optional[pd.DatetimeIndex] = None

    # Initial values for the martingale process
    start_values: Optional[np.ndarray] = None

    # Whether to adjust the mean of the martingale values
    mean_adjustment: bool = False

    # Precomputed mean values for adjustment
    mean_value: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True
        from_attributes = True

    def _safe_int(self, value: str) -> Optional[int]:
        """Safely convert a value to an integer, returning None if conversion fails."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def model_post_init(self, __context) -> None:
        """Initialize the scenario tree with an empty list."""
        os.makedirs(self.tree_name, exist_ok=True)

        db_path = os.path.join(self.tree_name, f"{self.tree_name}.db")
        csv_path = os.path.join(self.tree_name, f"{self.tree_name}.csv")

        if os.path.exists(db_path) and os.path.exists(csv_path):
            logger.info(
                f"Tree '{self.tree_name}' already exists. Loading existing instance."
            )
            tree = []
            with open(csv_path, "r") as f:
                for line in f:
                    try:
                        node_id_str, parent_id_str = line.strip().split(",")
                        node_id = self._safe_int(node_id_str)
                        parent_id = self._safe_int(parent_id_str)
                    except (AttributeError, ValueError, TypeError):
                        node_id, parent_id = None, None

                    tree.append((node_id, parent_id))
            self.tree = tree
        else:
            logger.info(
                f"No existing tree named '{self.tree_name}'. Creating a new one."
            )
            if os.path.exists(db_path):
                logger.warning(
                    f"Removing existing database file {db_path} to create a new one."
                )

                os.remove(db_path)

        logger.info(f"Database {self.tree_name}.db setup...")
        self._db_engine = create_engine(
            f"sqlite:///{self.tree_name}/{self.tree_name}.db",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        logger.info(
            f"Database engine created for {self.tree_name}.db. Creating tables..."
        )
        Base.metadata.create_all(self._db_engine)

        if self.model_input is None:
            raise ValueError(
                "Model input cannot be None. Please provide model input data."
            )
        if self.scenario_value is None:
            raise ValueError(
                "Scenario value cannot be None. Please provide scenario values."
            )
        if self.martingale_model is None:
            raise ValueError(
                "Martingale model cannot be None. Please provide a martingale model."
            )

        if self.tree is None or self.tree == []:
            self.tree = []
            self._add_root_node(
                node_id=0,
                model_input=self.model_input[:, -1],
                scenario=self.scenario_value,
                martingale=np.zeros(self.scenario_value.shape, dtype=float),
            )
        logger.info(f"Initialized ScenarioTree database: {self.tree_name}.db")

    @contextmanager
    def _get_session(self):
        """Context manager for database sessions with proper error handling."""
        if self._db_engine is None:
            raise ValueError("Database is not initialized.")

        session = Session(self._db_engine)
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def _serialize_array(self, arr: np.ndarray) -> bytes:
        """Serialize numpy array with validation."""
        if not isinstance(arr, np.ndarray):
            raise ValueError("Input must be a numpy array")

        buffer = io.BytesIO()
        np.save(buffer, arr, allow_pickle=True)
        return buffer.getvalue()

    def _deserialize_array(self, data: bytes) -> np.ndarray:
        """Deserialize numpy array with error handling."""
        try:
            buffer = io.BytesIO(data)
            return np.load(buffer, allow_pickle=True)
        except Exception as e:
            raise ValueError("Corrupted array data")

    def _node_exists_in_session(self, session: Session, node_id: int) -> bool:
        """Check if node exists within a session."""
        return session.query(DatabaseNode).filter_by(id=node_id).first() is not None

    def _insert_node(self, session: Session, node_data: TreeNode) -> None:
        """Insert a node into the database."""
        new_node = DatabaseNode(
            id=node_data.id,
            parent_id=node_data.parent_id,
            model_input=self._serialize_array(node_data.model_input),
            scenario=self._serialize_array(node_data.scenario),
            martingale=self._serialize_array(node_data.martingale),
            probability=node_data.probability,
            is_leaf=node_data.is_leaf,
        )
        session.add(new_node)

    def _add_root_node(
        self,
        node_id: int,
        model_input: np.ndarray,
        scenario: np.ndarray,
        martingale: np.ndarray,
    ) -> None:
        """Add the root node to the scenario tree."""
        # Validate input data
        node_data = TreeNode(
            id=node_id,
            parent_id=None,
            model_input=model_input,
            scenario=scenario,
            martingale=martingale,
            probability=1.0,
            is_leaf=True,
        )

        with self._get_session() as session:
            # Check if root already exists
            existing_root = (
                session.query(DatabaseNode).filter_by(parent_id=None).first()
            )
            if existing_root:
                raise ValueError("Root node already exists.")

            # Check if node ID already exists
            if self._node_exists_in_session(session, node_id):
                raise ValueError(f"Node with ID {node_id} already exists.")

            # Create root node
            self._insert_node(session, node_data)
            self.tree.append((node_id, None))

    def add_node(
        self,
        node_id: int,
        model_input: np.ndarray,
        scenario: np.ndarray,
        martingale: np.ndarray,
        parent_id: int,
        probability: float,
    ) -> None:
        """Add a node to the scenario tree."""

        # Validate input data
        node_data = TreeNode(
            id=node_id,
            parent_id=parent_id,
            model_input=model_input,
            scenario=scenario,
            martingale=martingale,
            probability=probability,
            is_leaf=True,
        )

        with self._get_session() as session:
            # Check if node ID already exists
            if self._node_exists_in_session(session, node_id):
                raise ValueError(f"Node with ID {node_id} already exists.")

            # Check if parent exists
            parent_node = session.query(DatabaseNode).filter_by(id=parent_id).first()
            if not parent_node:
                raise ValueError(f"Parent node with ID {parent_id} does not exist.")

            # Insert new node
            self._insert_node(session, node_data)

            # Update parent's leaf status
            if parent_node.is_leaf:
                parent_node.is_leaf = False
                session.add(parent_node)

            self.tree.append((node_id, parent_id))

    def get_node_data(self, node_id: int) -> Optional[TreeNode]:
        """Get node data as a detached object."""
        with self._get_session() as session:
            node = session.query(DatabaseNode).filter_by(id=node_id).first()
            if not node:
                return None

            return TreeNode(
                id=node.id,
                parent_id=node.parent_id,
                model_input=self._deserialize_array(node.model_input),
                scenario=self._deserialize_array(node.scenario),
                martingale=self._deserialize_array(node.martingale),
                probability=node.probability,
                is_leaf=node.is_leaf,
            )

    def get_children_ids(self, parent_id: int) -> List[int]:
        """Get child node IDs."""
        with self._get_session() as session:
            children = (
                session.query(DatabaseNode.id).filter_by(parent_id=parent_id).all()
            )
            return [child.id for child in children]

    def get_leaf_node_ids(self) -> List[int]:
        """Get all leaf node IDs."""
        with self._get_session() as session:
            leaves = session.query(DatabaseNode.id).filter_by(is_leaf=True).all()
            return [leaf.id for leaf in leaves]

    def get_path_to_root(self, node_id: int) -> List[int]:
        """Get path from node to root."""
        path = []
        current_id = node_id

        if node_id < 0:
            return path

        with self._get_session() as session:
            while current_id is not None:
                node = session.query(DatabaseNode).filter_by(id=current_id).first()
                if not node:
                    raise ValueError(f"Node {current_id} not found")

                path.append(current_id)
                current_id = node.parent_id

        return path

    def collect_scenario_path(self, leaf_id: int) -> List[np.ndarray]:
        """Collect scenario data from leaf to root."""
        path = self.get_path_to_root(leaf_id)
        scenarios = []

        with self._get_session() as session:
            for node_id in path:
                node = session.query(DatabaseNode).filter_by(id=node_id).first()
                if not node:
                    raise ValueError(f"Node {node_id} not found")
                scenarios.append(self._deserialize_array(node.scenario))
                del node

        return scenarios

    def node_exists(self, node_id: int) -> bool:
        """Check if a node exists."""
        with self._get_session() as session:
            return self._node_exists_in_session(session, node_id)

    def clear_tree(self) -> None:
        """Clear the entire tree."""
        with self._get_session() as session:
            session.query(DatabaseNode).delete()
            self.tree = []
            logger.info("Tree cleared successfully")

    def generate_scenario_tree(
        self,
        num_periods: int,
        num_scenarios_per_period: int,
        batch_size: int = 100,
        log_value: bool = False,
        returns: bool = True,
        solve_constraints: bool = True,
    ) -> int:
        """Generate a scenario tree with the given number of periods and scenarios per period."""

        def _generate_tree_helper(
            node: TreeNode, period: int, last_id: int, batch_count: int
        ) -> Tuple[int, int]:
            if period == num_periods:  # Base case: if we have reached the last period
                node.is_leaf = True
                return last_id, batch_count

            parent = node

            # Generate all the children values for the current node
            inputs, martingale, scenarios = self._generate_scenarios(
                parent_node=parent,
                num_scenarios=num_scenarios_per_period,
                log_value=log_value,
                returns=returns,
                solve_constraints=solve_constraints,
            )
            # Create a new child node for each scenario
            for i in range(num_scenarios_per_period):
                last_id += 1
                batch_count += 1

                if scenarios[i][1] > 1:
                    logger.warning(
                        f"Scenario value {scenarios[i]} is greater than 1. "
                        "This may indicate an issue with the scenario generation."
                    )

                # Create a new child node
                child_node = TreeNode(
                    is_leaf=True,
                    parent_id=parent.id,
                    model_input=inputs[i],
                    scenario=scenarios[i],
                    martingale=martingale[i],
                    probability=parent.probability * (1 / num_scenarios_per_period),
                    id=last_id,
                )

                # Add the child node to the parent using the new add_child method
                self.add_node(
                    node_id=child_node.id,
                    parent_id=parent.id,
                    model_input=child_node.model_input,
                    scenario=child_node.scenario,
                    martingale=child_node.martingale,
                    probability=child_node.probability,
                )

                # Make the recursive call to generate the next period
                last_id, batch_count = _generate_tree_helper(
                    child_node, period + 1, last_id, batch_count
                )
                if batch_count >= batch_size:
                    with self._get_session() as session:
                        session.commit()
                        batch_count = 0
                        logger.info(
                            f"Committed batch of {batch_size} nodes to database"
                        )

                del child_node  # Clean up to free memory

            return last_id, batch_count

        if len(self.tree) > 1:
            logger.warning(
                "Scenario tree already exists. Clearing it before generating a new one, keeping the root node."
            )
            root = self.get_node_data(self.tree[0][0])
            if root is None:
                raise ValueError(
                    "Root node does not exist. Please add a root node first."
                )
            self.clear_tree()
            self.tree = []
            self._add_root_node(
                node_id=root.id,
                model_input=root.model_input,
                scenario=root.scenario,
                martingale=root.martingale,
            )

        # Validate inputs
        if num_periods <= 0:
            raise ValueError("Number of periods must be greater than 0.")
        if num_scenarios_per_period <= 0:
            raise ValueError("Number of scenarios per period must be greater than 0.")
        if self.tree == [] or self.tree is None or len(self.tree) == 0:
            raise ValueError("Scenario tree is empty. Please generate the tree first.")

        current_node = self.get_node_data(self.tree[0][0])
        if current_node is None:
            raise ValueError("Root node does not exist. Please add a root node first.")

        last_id, _ = _generate_tree_helper(current_node, 0, current_node.id, 0)

        # Final commit for any remaining nodes
        with self._get_session() as session:
            session.commit()
            logger.info("Final commit completed")

        total_scenarios = num_scenarios_per_period**num_periods
        print(f"Last ID: {last_id}")
        print(f"Total scenarios generated: {total_scenarios}")

        return total_scenarios

    def _extract_scenario_path(self, parent_node: TreeNode) -> np.ndarray:
        path_ids = self.get_path_to_root(parent_node.id)[::-1]
        scenario_path = []
        for node_id in path_ids:
            node_data = self.get_node_data(node_id)
            if node_data is None:
                raise ValueError(f"Node {node_id} does not exist.")
            scenario_path.append(node_data.model_input)
        return np.array(scenario_path).T

    def _prepare_model_input(self, scenario_path: np.ndarray) -> np.ndarray:
        if self.model_input is None:
            raise ValueError("Model input cannot be None")
        return np.concatenate(
            (self.model_input[:, scenario_path.shape[1] :], scenario_path), axis=1
        )

    def _add_filtered_dates(
        self,
        existing_dates,
        num_points: int,
        start_hour: int = 9,
        end_hour: int = 17,
        avoid_weekends: bool = True,
        interval_hours: int = 1,
    ):
        from datetime import timedelta

        def is_valid_datetime(dt, start_hour, end_hour, avoid_weekends):
            if not (start_hour <= dt.hour <= end_hour):
                return False

            if avoid_weekends and dt.weekday() >= 5:
                return False

            return True

        if not isinstance(existing_dates, pd.DatetimeIndex):
            existing_dates = pd.to_datetime(existing_dates)

        last_date = existing_dates.max()

        new_dates = []
        current_date = last_date + timedelta(hours=interval_hours)

        while len(new_dates) < num_points:
            if is_valid_datetime(current_date, start_hour, end_hour, avoid_weekends):
                new_dates.append(current_date)

            current_date += timedelta(hours=interval_hours)

        all_dates = existing_dates.tolist() + new_dates
        # I want to return only the last 512 dates
        all_dates = all_dates[-512:]
        return pd.to_datetime(all_dates).sort_values()

    def _get_copula_means(self) -> List[float]:
        if self.martingale_model is None:
            raise ValueError(
                "Martingale model is required. Please provide a martingale model."
            )
        means = []
        if not isinstance(self.martingale_model, CopulaModel):
            logger.error(
                f"Martingale model is of type {type(self.martingale_model)}, expected CopulaModel."
            )
            return means
        if self.martingale_model.copula is None:
            raise ValueError("Copula is not set in the CopulaModel.")
        copula_dict = self.martingale_model.copula.to_dict()
        for i in range(len(copula_dict["univariates"])):
            means.append(copula_dict["univariates"][i]["loc"])
        return means

    def _generate_dates(self, scenario_path: np.ndarray) -> np.ndarray:
        return self._add_filtered_dates(
            existing_dates=self.dates,
            num_points=scenario_path.shape[1],
            start_hour=13,
            end_hour=19,
            avoid_weekends=True,
            interval_hours=1,
        )

    def _compute_means(self) -> np.ndarray:
        if self.model_input is None:
            raise ValueError("Model input cannot be None")
        if self.mean_value is not None:
            return self.mean_value
        if isinstance(self.martingale_model, CopulaModel):
            return np.array(self._get_copula_means())
        elif isinstance(self.martingale_model, BaseMartingaleModel):
            return np.asarray(self.martingale_model.mu)
        else:
            return np.zeros_like(self.model_input[:, 0])

    def _generate_prediction(
        self,
        model_input: np.ndarray,
        previous_scenario_value: np.ndarray,
        means: np.ndarray,
        dates,
        log_value: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.predictive_model is None:
            logger.info("No predictive model provided. Using EWMA for predictions.")
            df = pd.DataFrame(model_input).T
            ewma = df.ewm(com=0.5, min_periods=100).mean().dropna()
            prediction = ewma.to_numpy().flatten()
            if prediction.size == 0:
                raise ValueError(
                    f"Failed to generate prediction. Input shape: {model_input.shape}"
                )

            prediction = np.clip(prediction, -10, 10)

            if self.mean_adjustment:
                post_processed_prediction = (
                    np.exp(prediction) + previous_scenario_value + means
                )
            else:
                post_processed_prediction = np.exp(prediction) + previous_scenario_value

            post_processed_prediction = np.maximum(post_processed_prediction, 1e-8)
        else:
            if not hasattr(self.predictive_model, "predict"):
                raise ValueError("Predictive model must have a 'predict' method.")
            if self.cols_name is None or len(self.cols_name) == 0:
                raise ValueError("Columns must be provided for the predictive model.")
            if model_input.shape[0] != len(self.cols_name):
                raise ValueError(
                    f"Model input shape {model_input.shape} does not match columns length {len(self.cols_name)}."
                )

            df = pd.DataFrame(model_input).T
            df.columns = self.cols_name
            df[self.predictive_model.timestamp_column] = dates

            for col in self.cols_name:
                if np.any(~np.isfinite(df[col])):
                    logger.warning(
                        f"Column {col} contains non-finite values. Cleaning..."
                    )
                    df[col] = df[col].fillna(method="ffill").fillna(0)  # type: ignore
                    df[col] = np.clip(df[col], -10, 10)

            prediction = self.predictive_model.predict(df)
            predictions = (
                prediction[[f"{col}_prediction" for col in self.cols_name]]
                .to_numpy()
                .flatten()
            )
            predictions = np.clip(predictions, -10, 10)
            if self.mean_adjustment:
                if log_value:
                    post_processed_prediction = (
                        np.exp(predictions) * self.start_values + means
                    )
                else:
                    post_processed_prediction = predictions + self.start_values + means
            else:
                if log_value:
                    post_processed_prediction = np.exp(predictions) * self.start_values
                else:
                    post_processed_prediction = predictions + self.start_values

            post_processed_prediction = np.maximum(post_processed_prediction, 1e-8)
            prediction = predictions

            if np.any(~np.isfinite(prediction)):
                logger.error(f"Prediction contains non-finite values: {prediction}")
                prediction = np.nan_to_num(
                    prediction, nan=0.0, posinf=10.0, neginf=-10.0
                )

            if np.any(~np.isfinite(post_processed_prediction)):
                logger.error(
                    f"Post-processed prediction contains non-finite values: {post_processed_prediction}"
                )
                post_processed_prediction = np.nan_to_num(
                    post_processed_prediction,
                    nan=previous_scenario_value[-1],
                    posinf=1e6,
                    neginf=1e-8,
                )

        return prediction, post_processed_prediction

    def _generate_martingale_values(self, num_scenarios: int) -> np.ndarray:
        if self.martingale_model is None:
            raise ValueError("Martingale model is required.")
        if num_scenarios - 1 <= 0:
            logger.warning("Number of scenarios minus one is zero. Returning zeros.")
            assert (
                self.cols_name and len(self.cols_name) > 0
            ), "Columns must be provided."
            return np.zeros((1, len(self.cols_name)))
        martingale_values = self.martingale_model.sample(num_scenarios - 1)
        if isinstance(martingale_values, pd.DataFrame):
            martingale_values = martingale_values.to_numpy()
        return martingale_values

    def _constraints_check(
        self,
        prediction: np.ndarray,
        martingale_value: np.ndarray,
        solve_constraints: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Check and solve constraints for the generated scenarios."""
        if self.constraints is None or len(self.constraints) == 0:
            return np.asarray(prediction), np.asarray(martingale_value)

        for i in range(martingale_value.shape[0]):
            final_values = martingale_value[i] + prediction
            for constraint in self.constraints:
                for j in range(len(final_values)):
                    if not constraint.evaluate(final_values[j]):
                        if solve_constraints:
                            if not constraint.evaluate(prediction[j]):
                                fixed_value = constraint.solve(
                                    prediction[j], martingale_value[i][j]
                                )
                                prediction[j] = (
                                    fixed_value if fixed_value != 0 else 1e-8
                                )
                            fixed_value = constraint.solve(
                                prediction[j], martingale_value[i][j]
                            )

                            final_values[j] = fixed_value
                            martingale_value[i][j] = fixed_value - prediction[j]
                        else:
                            raise ValueError(
                                f"Constraint violation for variable {j} in scenario {i}: "
                                f"{final_values[j]} does not satisfy {constraint}"
                            )

        return np.asarray(prediction), np.asarray(martingale_value)

    def _generate_scenarios(
        self,
        parent_node: TreeNode,
        num_scenarios: int,
        log_value: bool = False,
        returns: bool = True,
        solve_constraints: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the scenarios in the scenario tree based on parent node."""
        if self.model_input is None:
            raise ValueError("Model input cannot be None")

        scenario_path = self._extract_scenario_path(parent_node)
        model_input = self._prepare_model_input(scenario_path)
        previous_scenario_value = parent_node.scenario
        dates = self._generate_dates(scenario_path)
        means = self._compute_means()

        if previous_scenario_value is None:
            raise ValueError("Previous scenario value cannot be None")

        if np.any(previous_scenario_value <= 0):
            logger.warning(
                f"Previous scenario value contains non-positive values: {previous_scenario_value}"
            )
            previous_scenario_value = np.maximum(previous_scenario_value, 1e-8)

        prediction, post_processed_prediction = self._generate_prediction(
            model_input, previous_scenario_value, means, dates, log_value
        )

        if np.any(~np.isfinite(prediction)):
            logger.warning(f"Prediction contains non-finite values: {prediction}")
            prediction = np.nan_to_num(prediction, nan=0.0, posinf=1e6, neginf=-1e6)

        if np.any(~np.isfinite(post_processed_prediction)):
            logger.warning(
                f"Post-processed prediction contains non-finite values: {post_processed_prediction}"
            )
            post_processed_prediction = np.nan_to_num(
                post_processed_prediction,
                nan=previous_scenario_value[-1],
                posinf=1e6,
                neginf=1e-8,
            )

        if log_value:
            new_scenario_values = [prediction]
        else:
            new_scenario_values = [post_processed_prediction]

        martingale_values = self._generate_martingale_values(num_scenarios)

        if self.constraints:
            new_scenario_values[0] = np.nan_to_num(
                new_scenario_values[0], nan=0.0, posinf=1e6, neginf=1e-8
            )
            martingale_values = np.nan_to_num(
                martingale_values, nan=0.0, posinf=1e6, neginf=-1e6
            )
            new_scenario_values[0], martingale_values = self._constraints_check(
                new_scenario_values[0], martingale_values, solve_constraints
            )

        new_model_inputs = [prediction]

        for i in range(num_scenarios - 1):
            martingale_value = martingale_values[i]
            new_value = martingale_value + new_scenario_values[0]

            new_value = np.maximum(new_value, 1e-8)
            if log_value:
                delta = np.log(new_value / previous_scenario_value)
            else:
                delta = new_value - previous_scenario_value

            # Additional check for delta
            if np.any(~np.isfinite(delta)):
                logger.warning(f"Delta contains non-finite values: {delta}")
                delta = np.nan_to_num(delta, nan=0.0, posinf=10.0, neginf=-10.0)
            if returns:
                new_model_inputs.append(delta)
            else:
                new_model_inputs.append(model_input[:, -1] + delta)

            new_scenario_values.append(new_value)

        return (
            np.array(new_model_inputs),
            np.insert(martingale_values, 0, 0, axis=0),
            np.array(new_scenario_values),
        )

    def save_leaf_distribution(
        self,
        filename: str,
        tickers: List[str],
    ) -> None:
        """
        Save the distribution of leaf nodes to a file.

        Args:
            filename (str): The name of the file to save the distribution.
            tickers (List[str]): List of tickers for the leaf nodes.
        """
        with self._get_session() as session:
            leaves = session.query(DatabaseNode).filter_by(is_leaf=True).all()
            if not leaves:
                logger.warning("No leaf nodes found.")
                return

            df = pd.DataFrame()
            for i, ticker in enumerate(tickers):
                # Create a new column for each ticker
                leaf_values = []
                for node in leaves:
                    if node.scenario is not None:
                        # Deserialize the scenario data
                        scenario = self._deserialize_array(node.scenario)
                        # Ensure the scenario is a 1D array
                        if scenario.ndim > 1:
                            raise ValueError("Scenario data must be a 1D array.")
                        # Append the value for the current ticker
                        if i < len(scenario):
                            leaf_values.append(scenario[i])
                        else:
                            leaf_values.append(np.nan)
                # Convert to numpy array
                leaf_values = np.array(leaf_values)
                df[ticker] = leaf_values

            df.to_csv(filename, index=False)

    def view_scenario_tree(
        self, tickers: List[str], figsize=(16, 9), bins: Union[str, int] = "auto"
    ) -> None:
        """
        Visualize the scenario tree for each variable in separate figures.
        Each figure has two parts: scenario paths over time (left) and a KDE of final states (right).

        Args:
            figsize (tuple): Figure size for each plot (width, height).
        """
        try:
            leaves = self.get_leaf_node_ids()
            if not leaves:
                raise ValueError("No leaf nodes available. Generate the tree first.")
            scenarios = []
            for leaf_id in leaves:
                path = self.collect_scenario_path(leaf_id)[::-1]
                if path:
                    scenarios.append(path)
            if not scenarios:
                raise ValueError("No scenarios available. Generate the tree first.")

            scenario_data = np.array(
                [
                    [
                        val if isinstance(val, np.ndarray) else val.numpy()
                        for val in path
                    ]
                    for path in scenarios
                ]
            )
            num_scenarios, num_periods, num_variables = scenario_data.shape
            time_steps = list(range(1, num_periods + 1))

            for var_idx in range(num_variables):
                var_data = scenario_data[:, :, var_idx]

                fig = plt.figure(figsize=figsize)
                gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

                ax1 = fig.add_subplot(gs[0, 0])
                colors = cm.get_cmap("viridis", num_scenarios)

                for i in range(num_scenarios - 1):
                    ax1.plot(
                        time_steps, var_data[i], marker="o", alpha=0.6, color=colors(i)
                    )
                ax1.set_title("states")
                ax1.set_xlabel("stage, time")
                ax1.set_ylabel("states")
                ax1.set_xticks(time_steps)

                y_min, y_max = ax1.get_ylim()

                ax2 = fig.add_subplot(gs[0, 1])
                final_states = var_data[:, -1]

                ax2.hist(
                    final_states,
                    bins=bins,
                    color="blue",
                    alpha=0.7,
                    orientation="horizontal",
                    density=True,
                )

                # Align the y-axis with ax1
                ax2.set_ylim(y_min, y_max)

                # Dynamically set the x-axis (frequency) limit based on the histogram
                freq_max = ax2.get_xlim()[1]  # Get the maximum frequency
                ax2.set_xlim(0, freq_max * 1.1)
                ax2.set_xlabel("Density")
                ax2.set_ylabel("")

                plt.suptitle(f"Scenario Tree {tickers[var_idx]}", y=1.02)
                plt.tight_layout(rect=(0, 0, 1, 0.98))
                plt.show()

        except ImportError as e:
            print(e)
            print(
                "Matplotlib or Seaborn not available. Displaying text representation instead:"
            )
            print(str(self))

    def save_to_file(self) -> None:
        """
        Save the scenario tree to a file.

        Args:
            filename (str): The name of the file to save the scenario tree.
        """
        path = os.path.join(self.tree_name, f"{self.tree_name}.csv")
        if os.path.exists(path):
            logger.warning(f"File {self.tree_name} already exists. Overwriting it.")

        with open(path, "w") as f:
            for node_id, parent_id in self.tree:
                f.write(f"{node_id},{parent_id}\n")
        logger.info(f"Scenario tree saved to {self.tree_name}")

    def num_nodes(self) -> int:
        """Return the number of nodes in the scenario tree."""
        return len(self.tree)

    def predecessors(self, node_id: int) -> List[int]:
        """Return the predecessors of a given node."""
        parent_map = {child: parent for child, parent in self.tree}
        predecessors = []
        current = parent_map.get(node_id)

        while current is not None:
            predecessors.append(current)
            current = parent_map.get(current)

        return predecessors

    def num_scenarios(self) -> int:
        """Return the total number of scenarios in the scenario tree."""
        with self._get_session() as session:
            return (
                session.query(DatabaseNode).filter(DatabaseNode.is_leaf == True).count()
            )

    def tree_depth(self) -> int:
        """Return the depth of the scenario tree."""
        with self._get_session() as session:
            root = session.query(DatabaseNode).filter_by(parent_id=None).first()
            if not root:
                raise ValueError("Root node does not exist.")
            max_depth = 0
            nodes_to_visit = [(root, 1)]
            while nodes_to_visit:
                current_node, depth = nodes_to_visit.pop()
                max_depth = max(max_depth, depth)
                children = (
                    session.query(DatabaseNode)
                    .filter_by(parent_id=current_node.id)
                    .all()
                )
                for child in children:
                    nodes_to_visit.append((child, depth + 1))
            return max_depth

    def get_every_node_probability(self) -> List[float]:
        """Return the probability of each node in the scenario tree."""
        probabilities = []
        with self._get_session() as session:
            nodes = session.query(DatabaseNode).all()
            for node in nodes:
                probabilities.append(node.probability)
        return probabilities

    def get_node_depth(self, node_id: int) -> int:
        """Return the depth of a specific node in the scenario tree."""
        if not self.node_exists(node_id):
            raise ValueError(f"Node with ID {node_id} does not exist.")

        path_to_root = self.get_path_to_root(node_id)
        if not path_to_root:
            raise ValueError(f"No path found to root for node {node_id}.")
        return len(path_to_root)
