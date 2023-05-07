#include <ubinos.h>

#if (UBINOS__BSP__BOARD_MODEL == UBINOS__BSP__BOARD_MODEL__NUCLEOF207ZG)

#include <ubinos/bsp/arch.h>

#include "main.h"

/**
 * @brief UART MSP Initialization
 *        This function configures the hardware resources used in this example:
 *           - Peripheral's clock enable
 *           - Peripheral's GPIO Configuration
 *           - DMA configuration for transmission request by peripheral
 *           - NVIC configuration for DMA interrupt request enable
 * @param huart: UART handle pointer
 * @retval None
 */
void HAL_UART_MspInit(UART_HandleTypeDef *huart)
{
    GPIO_InitTypeDef GPIO_InitStruct;

    /*##-1- Enable peripherals and GPIO Clocks #################################*/
    /* Enable GPIO clock */
    DTTY_STM32_UART_TX_GPIO_CLK_ENABLE();
    DTTY_STM32_UART_RX_GPIO_CLK_ENABLE();
    /* Enable DTTY_STM32_UART clock */
    DTTY_STM32_UART_CLK_ENABLE();

    /*##-2- Configure peripheral GPIO ##########################################*/
    /* UART TX GPIO pin configuration  */
    GPIO_InitStruct.Pin = DTTY_STM32_UART_TX_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FAST;
    GPIO_InitStruct.Alternate = DTTY_STM32_UART_TX_AF;

    HAL_GPIO_Init(DTTY_STM32_UART_TX_GPIO_Port, &GPIO_InitStruct);

    /* UART RX GPIO pin configuration  */
    GPIO_InitStruct.Pin = DTTY_STM32_UART_RX_Pin;
    GPIO_InitStruct.Alternate = DTTY_STM32_UART_RX_AF;

    HAL_GPIO_Init(DTTY_STM32_UART_RX_GPIO_Port, &GPIO_InitStruct);

    /* NVIC configuration for USART TC interrupt */
    HAL_NVIC_SetPriority(DTTY_STM32_UART_IRQn, NVIC_PRIO_MIDDLE, 0);
    HAL_NVIC_EnableIRQ(DTTY_STM32_UART_IRQn);
}

/**
 * @brief UART MSP De-Initialization
 *        This function frees the hardware resources used in this example:
 *          - Disable the Peripheral's clock
 *          - Revert GPIO, DMA and NVIC configuration to their default state
 * @param huart: UART handle pointer
 * @retval None
 */
void HAL_UART_MspDeInit(UART_HandleTypeDef *huart)
{
    /*##-1- Reset peripherals ##################################################*/
    DTTY_STM32_UART_FORCE_RESET();
    DTTY_STM32_UART_RELEASE_RESET();

    /*##-2- Disable peripherals and GPIO Clocks #################################*/
    /* Configure UART Tx as alternate function  */
    HAL_GPIO_DeInit(DTTY_STM32_UART_TX_GPIO_Port, DTTY_STM32_UART_TX_Pin);
    /* Configure UART Rx as alternate function  */
    HAL_GPIO_DeInit(DTTY_STM32_UART_RX_GPIO_Port, DTTY_STM32_UART_RX_Pin);

    /*##-3- Disable the NVIC for USART TC ###########################################*/
    HAL_NVIC_DisableIRQ(DTTY_STM32_UART_IRQn);
}

#endif /* (UBINOS__BSP__BOARD_MODEL == UBINOS__BSP__BOARD_MODEL__NUCLEOF207ZG) */

